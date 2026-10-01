//! Transmit-side loudness NORMALISER: a slow level follower with a per-block
//! limiter.
//!
//! Deliberately not called a compressor, because it is not one. A compressor
//! works on the dynamics inside a signal — it pulls the loud syllables down and
//! so reduces the crest factor. This computes ONE gain from a slow average and
//! glides it at a few dB per second, which levels whole transmissions and whole
//! handsets against each other and leaves the crest alone (measured on a real
//! device: 16.0, 15.8, 14.3, 16.4 dB across four settings — unchanged). The name
//! is a promise about behaviour; the wrong one would have future readers expect
//! dynamics it does not do.
//!
//! Why this exists: measured through the `#echo` bot on 2026-09-30/10-01, iOS
//! handsets transmit around -30 dBFS RMS while Android sits near -18 — a ~12 dB
//! platform gap that no receive-side control can undo (the WebRTC mixer's
//! limiter absorbs anything pushed past nominal, and `Helper.setVolume` is
//! clamped at unity anyway). Plain make-up gain cannot fix it either: a real
//! user measured -27.2 dBFS RMS with peaks already at -2.9 dBFS, i.e. 24 dB of
//! crest and 3 dB of headroom. Raising perceived loudness there means reducing
//! the crest factor, which is what the compressor in a handheld radio does — and which this
//! stage explicitly does NOT; see the note above.
//!
//! Design, in order of what it protects:
//!
//! * **It can never clip.** The capture hook hands us a whole 10 ms block
//!   before anything is written back, so we know the block's true peak in
//!   advance — free lookahead. The gain is capped per block at
//!   `ceiling / block_peak`, so no sample can cross the ceiling and there is no
//!   need for the soft clipping that makes cheap compressors sound harsh.
//! * **It must not pump.** The level follower is slow and the gain glides at a
//!   bounded rate (dB per second), so it cannot move audibly inside a phrase.
//!   Only the ceiling cap may act fast, and only downwards, which is a limiter
//!   and is inaudible when brief.
//! * **It must leave a loud talker alone.** The gain is clamped at `>= 1.0` and
//!   driven by the distance to the target, so a signal already at or above the
//!   target is passed through.
//! * **Silence must not wind it up.** Blocks below the noise floor do not feed
//!   the follower, otherwise the gap between phrases would ask for maximum gain
//!   and the next word would arrive as a bang.
//!
//! Samples are f32 in PCM16 amplitude space (±32768), the format the capture
//! hook delivers on both platforms.

/// Full-scale amplitude of the sample format used by the capture hook.
pub const FULL_SCALE: f32 = 32768.0;

/// Amplitude for a dBFS value, e.g. `dbfs(-20.0) == 3276.8`.
pub fn dbfs(db: f32) -> f32 {
    FULL_SCALE * 10f32.powf(db / 20.0)
}

#[derive(Clone, Copy, Debug)]
pub struct NormalizerSettings {
    pub enabled: bool,
    /// Loudness aimed for, as an amplitude in PCM16 space. -20 dBFS by default,
    /// which is where a healthy Android transmission already sits.
    pub target_rms: f32,
    /// No sample may exceed this. NOT user-adjustable: it is the anti-clipping
    /// rail, and the whole no-distortion argument rests on it.
    pub ceiling: f32,
    /// Upper bound on make-up gain. Without it a near-silent input would be
    /// lifted until its noise floor became the signal.
    pub max_gain: f32,
}

impl Default for NormalizerSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            target_rms: dbfs(-20.0),
            ceiling: dbfs(-1.0),
            max_gain: 8.0,
        }
    }
}

/// Blocks quieter than this never update the level follower.
const NOISE_FLOOR: f32 = 103.0; // ≈ -50 dBFS
/// Seconds for the follower to track a rise / a fall in level.
const RMS_RISE_SEC: f32 = 0.2;
const RMS_FALL_SEC: f32 = 0.8;
/// How much of the make-up gain the input has earned, by how speech-shaped it is.
///
/// Found the hard way: on an Android emulator the virtual mic delivered a steady
/// hum, the follower took it for signal and parked it neatly on the target, and
/// `#echo` played back a -20 dBFS drone. A steady tone is the *ideal* input for a
/// level follower, which is exactly the trap.
///
/// Crest factor separates the two, but not cleanly enough for a hard threshold:
/// measured through the bot, real speech ran 14–16 dB on Android and 17–24 on
/// iOS, while that emulator hum reached 8.5 dB — and a pure tone sits at 3. A
/// line drawn between them would be a guess, and a signal just under it would
/// fall off a cliff. So the gain is scaled instead: none below [`CREST_NONE_DB`],
/// all of it above [`CREST_FULL_DB`], proportional between. The less it looks
/// like a voice, the less help it gets, and nothing has a cliff to fall off.
/// Note the window matters — a single 10 ms block of a sustained vowel is almost
/// a tone too, so this is measured across syllables, pauses included.
const CREST_NONE_DB: f32 = 8.0;
const CREST_FULL_DB: f32 = 14.0;
/// Seconds for the windowed peak to fall back; it rises instantly.
const PEAK_FALL_SEC: f32 = 1.0;

/// How fast the applied gain may travel, in dB per second.
const GAIN_UP_DB_PER_SEC: f32 = 9.0;
const GAIN_DOWN_DB_PER_SEC: f32 = 18.0;

/// A voice through a phone microphone is never this quiet. Measured in `#echo`
/// across five handsets, the quietest real transmission was -31.2 dBFS RMS and
/// the quietest platform average -30; this sits 9 dB below that, far enough not
/// to touch a soft talker and far above the room noise that used to pass for
/// one. Without it, `target / level` on an empty room asks for `max_gain` and
/// gets it, which is how the readout could reach +18 dB with nobody speaking.
const VOICE_FLOOR: f32 = 103.0 * 3.17; // ~ -40 dBFS

/// Blocks the crest window needs before its reading means anything.
///
/// `window_mean_sq` and `peak_level` both start empty, so on the first block
/// after a reset the "crest across syllables and pauses" is just the crest of
/// one 10 ms block — and white noise measures ~12 dB that way, which is most of
/// the way to [`CREST_FULL_DB`]. Switching the level mode calls `reset`, so this
/// blind spot landed exactly where a user would notice it: the gain ran to
/// maximum on room noise right after the switch and the ratchet kept it there.
/// Half of [`PEAK_FALL_SEC`] is the shortest window that is actually a window.
const CREST_WARMUP_BLOCKS: u32 = 50;

/// How long a silence is held before the gain is given back. Long enough to
/// cover the pause between words (300 ms in the test fixture, and a breath is
/// not much more), short enough that a finished transmission visibly returns to
/// unity rather than latching the last value it reached.
const RELEASE_HOLD_SEC: f32 = 1.0;

pub struct Normalizer {
    settings: NormalizerSettings,
    /// Smoothed RMS of speech-level blocks, in PCM16 amplitude.
    level: f32,
    /// Windowed peak, for the crest test. Rises instantly, falls slowly.
    peak_level: f32,
    /// Mean square over the same window INCLUDING the pauses between words.
    /// Separate from `level` on purpose: `level` skips silent blocks, because a
    /// gain target must follow the voice and not the gaps — but a crest measured
    /// against that would miss the very gaps that make speech spiky, and would
    /// not be comparable to the whole-transmission crest `#echo` reports (which
    /// is what the threshold was calibrated against: 14–24 dB for real speech
    /// across three handsets, 3.8–8.5 dB for the emulator hum).
    window_mean_sq: f32,
    /// Gain actually applied at the end of the previous block.
    gain: f32,
    /// Blocks processed since the last reset, capped — see [`CREST_WARMUP_BLOCKS`].
    blocks_seen: u32,
    /// Consecutive blocks with no voice in them, for the release below.
    quiet_blocks: u32,
}

impl Normalizer {
    pub fn new(settings: NormalizerSettings) -> Self {
        Self {
            settings,
            level: 0.0,
            peak_level: 0.0,
            window_mean_sq: 0.0,
            gain: 1.0,
            blocks_seen: 0,
            quiet_blocks: 0,
        }
    }

    pub fn set_settings(&mut self, settings: NormalizerSettings) {
        self.settings = settings;
    }

    pub fn settings(&self) -> NormalizerSettings {
        self.settings
    }

    /// Gain applied at the end of the last processed block — for diagnostics.
    pub fn current_gain(&self) -> f32 {
        self.gain
    }

    pub fn reset(&mut self) {
        self.level = 0.0;
        self.peak_level = 0.0;
        self.window_mean_sq = 0.0;
        self.gain = 1.0;
        self.blocks_seen = 0;
        self.quiet_blocks = 0;
    }

    /// Crest factor of the follower's window, in dB. Below
    /// [`MIN_SPEECH_CREST_DB`] the input has no speech dynamics.
    pub fn crest_db(&self) -> f32 {
        let window_rms = self.window_mean_sq.max(0.0).sqrt();
        if window_rms <= 0.0 || self.peak_level <= 0.0 {
            return 0.0;
        }
        20.0 * (self.peak_level / window_rms).log10()
    }

    /// Processes one block in place. Returns the gain applied at the end of it.
    pub fn process(&mut self, samples: &mut [f32], sample_rate: u32) -> f32 {
        if !self.settings.enabled || samples.is_empty() || sample_rate == 0 {
            // Return exactly 1.0, not the last gain we happened to hold: the
            // bridges use `gain != 1.0` to decide whether to write the block
            // back, so a stale value would make them copy an untouched buffer
            // and the guard would stop meaning "this stage changed something".
            if !self.settings.enabled {
                self.gain = 1.0;
                self.level = 0.0;
                self.peak_level = 0.0;
                self.window_mean_sq = 0.0;
                self.blocks_seen = 0;
                self.quiet_blocks = 0;
            }
            return 1.0;
        }

        // WebRTC's APM always hands over exactly 10 ms, whatever rate it runs at,
        // and the rate the bridge reports is NOT reliable: on an Android emulator
        // it said 48 kHz while delivering 160-sample blocks, i.e. 16 kHz. Trusting
        // it there would have made every time constant three times too fast. The
        // block IS the clock.
        let block_sec = 0.01;
        let _ = sample_rate;
        let mut peak = 0.0f32;
        let mut sum_sq = 0.0f64;
        for sample in samples.iter() {
            let magnitude = sample.abs();
            if magnitude > peak {
                peak = magnitude;
            }
            sum_sq += (*sample as f64) * (*sample as f64);
        }
        let rms = (sum_sq / samples.len() as f64).sqrt() as f32;

        // Window mean square, pauses included — see `window_mean_sq`.
        let window_alpha = 1.0 - (-block_sec / PEAK_FALL_SEC).exp();
        let block_mean_sq = rms * rms;
        if self.window_mean_sq <= 0.0 {
            self.window_mean_sq = block_mean_sq;
        } else {
            self.window_mean_sq += (block_mean_sq - self.window_mean_sq) * window_alpha;
        }

        // Windowed peak for the crest test: instant attack, slow release.
        if peak > self.peak_level {
            self.peak_level = peak;
        } else {
            let fall = (-block_sec / PEAK_FALL_SEC).exp();
            self.peak_level *= fall;
        }

        self.blocks_seen = self.blocks_seen.saturating_add(1);

        // Only real audio moves the follower; the pauses between words must not.
        if peak >= NOISE_FLOOR {
            let tau = if rms > self.level { RMS_RISE_SEC } else { RMS_FALL_SEC };
            let alpha = 1.0 - (-block_sec / tau).exp();
            if self.level <= 0.0 {
                self.level = rms;
            } else {
                self.level += (rms - self.level) * alpha;
            }
        }

        // Where we would like the gain to be. Never below unity: pulling a loud
        // talker down is the ceiling's job, not the follower's.
        let desired = if self.level > 0.0 {
            (self.settings.target_rms / self.level).clamp(1.0, self.settings.max_gain)
        } else {
            1.0
        };

        // Is there a voice here at all? Three things have to hold, and every one
        // of them was once a way for this stage to hand its make-up gain to
        // something that was not a voice: audio above the noise floor, a level a
        // voice could plausibly have, and a crest window old enough to have
        // measured anything. The honest limit is the middle one — a room noisy
        // enough to sit above VOICE_FLOOR still reads as a quiet talker, and
        // only a real VAD would separate those. RNNoise has one, but this stage
        // runs with the AI filter off too, so it cannot be used here.
        let voice = peak >= NOISE_FLOOR
            && self.level >= VOICE_FLOOR
            && self.blocks_seen >= CREST_WARMUP_BLOCKS;
        if voice {
            self.quiet_blocks = 0;
        } else {
            self.quiet_blocks = self.quiet_blocks.saturating_add(1);
        }

        // Scale the lift by how speech-shaped the window is, so hum, whine and
        // emulator noise get none of it. Only the RISE is scaled: falling is
        // always allowed, and the floor is the gain already applied, so a
        // sustained vowel mid-sentence cannot drop the gain out from under the
        // speaker.
        let allowance = ((self.crest_db() - CREST_NONE_DB)
            / (CREST_FULL_DB - CREST_NONE_DB))
            .clamp(0.0, 1.0);
        let earned = 1.0 + (desired - 1.0) * allowance;
        let desired = if voice {
            earned.max(self.gain.min(desired))
        } else if (self.quiet_blocks as f32) * block_sec <= RELEASE_HOLD_SEC {
            // Hold through the pause between words: that floor is the whole
            // reason the ratchet above exists, and it must survive a breath.
            self.gain
        } else {
            // Past the hold, give it back. Without this the floor was absolute:
            // `level` never updates below the noise floor, so nothing could ever
            // lower `desired` again and whatever gain a transmission reached was
            // the gain for the rest of the session — which is what the readout
            // was faithfully showing when it appeared stuck.
            1.0
        };

        // Glide, so the gain cannot move audibly within a phrase.
        let max_up = 10f32.powf(GAIN_UP_DB_PER_SEC * block_sec / 20.0);
        let max_down = 10f32.powf(-GAIN_DOWN_DB_PER_SEC * block_sec / 20.0);
        let gain = desired
            .min(self.gain * max_up)
            .max(self.gain * max_down)
            .max(1.0);

        // The rail. We hold the whole block, so the limit is exact rather than
        // predictive. It has to bound EVERY sample, not just the block's final
        // gain: the ramp below starts from the previous block's gain, so a gain
        // wound up during quiet speech would be applied to the first samples of
        // a sudden shout. That is how the first version of this clipped at
        // +10.6 dBFS, and why `a_transient_after_quiet_speech_does_not_clip`
        // exists.
        let rail = if peak > 0.0 { self.settings.ceiling / peak } else { f32::INFINITY };

        // Ramp across the block so a change of gain never lands as a step, and
        // ride the rail wherever the ramp would cross it — which is exactly what
        // a limiter does, and it releases afterwards via the glide above.
        let start = self.gain;
        let span = samples.len() as f32;
        let mut applied = start.min(rail);
        for (index, sample) in samples.iter_mut().enumerate() {
            let t = (index as f32 + 1.0) / span;
            applied = (start + (gain - start) * t).min(rail);
            *sample *= applied;
        }

        self.gain = applied;
        applied
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: u32 = 48_000;
    const BLOCK: usize = 480; // 10 ms

    fn settings(enabled: bool) -> NormalizerSettings {
        NormalizerSettings { enabled, ..Default::default() }
    }

    /// A sine block generator keeping phase across calls.
    struct Tone {
        phase: f32,
        hz: f32,
        amplitude: f32,
    }

    impl Tone {
        fn new(hz: f32, rms_dbfs: f32) -> Self {
            // For a sine, peak = rms * sqrt(2).
            Self { phase: 0.0, hz, amplitude: dbfs(rms_dbfs) * std::f32::consts::SQRT_2 }
        }

        fn block(&mut self) -> Vec<f32> {
            (0..BLOCK)
                .map(|_| {
                    let value = self.phase.sin() * self.amplitude;
                    self.phase += 2.0 * std::f32::consts::PI * self.hz / RATE as f32;
                    value
                })
                .collect()
        }
    }

    fn peak_dbfs(samples: &[f32]) -> f32 {
        let peak = samples.iter().fold(0.0f32, |acc, s| acc.max(s.abs()));
        20.0 * (peak / FULL_SCALE).log10()
    }

    /// Speech-like block source: noise under a slow syllabic envelope, which
    /// lands at a crest factor near real speech (13–18 dB) rather than a sine's
    /// 3 dB. Deterministic, so the numbers below are reproducible.
    struct Speech {
        seed: u32,
        t: f32,
        rms_target: f32,
    }

    impl Speech {
        fn new(rms_dbfs: f32) -> Self {
            Self { seed: 0x1234_5678, t: 0.0, rms_target: dbfs(rms_dbfs) }
        }

        fn next_noise(&mut self) -> f32 {
            self.seed = self.seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((self.seed >> 8) as f32 / 8_388_608.0) - 1.0
        }

        fn block(&mut self) -> Vec<f32> {
            (0..BLOCK)
                .map(|_| {
                    // Syllables at 3.5 Hz, and — the part that actually makes
                    // speech spiky — a pause between words: 600 ms of talking,
                    // 300 ms of silence.
                    let word = (self.t % 0.9) < 0.6;
                    let env = if word {
                        (0.5 - 0.5 * (2.0 * std::f32::consts::PI * 3.5 * self.t).cos()).powi(2)
                    } else {
                        0.0
                    };
                    self.t += 1.0 / RATE as f32;
                    self.next_noise() * env * self.rms_target * 4.0
                })
                .collect()
        }
    }

    /// Room noise: continuous, no syllabic envelope and no pauses. Its settled
    /// crest sits near 12 dB, which is most of the way to [`CREST_FULL_DB`] — so
    /// the crest gate alone never kept it out, and after a reset, when the window
    /// is one block long, it looks spikier still.
    struct Noise {
        seed: u32,
        amplitude: f32,
    }

    impl Noise {
        fn new(rms_dbfs: f32) -> Self {
            // Uniform noise on [-1, 1] has an RMS of 1/sqrt(3).
            Self { seed: 0x5EED_1234, amplitude: dbfs(rms_dbfs) * 3f32.sqrt() }
        }

        fn block(&mut self) -> Vec<f32> {
            (0..BLOCK)
                .map(|_| {
                    self.seed = self.seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    (((self.seed >> 8) as f32 / 8_388_608.0) - 1.0) * self.amplitude
                })
                .collect()
        }
    }

    /// Not an assertion — a measurement, printed with `--nocapture`: how far the
    /// target can usefully be pushed before `max_gain` saturates, for the quiet
    /// handset the compressor exists for. It does NOT say where it starts to
    /// sound squashed; a first attempt tried to proxy that by counting blocks
    /// where the limiter bound, and the detector was wrong (100 % at every
    /// target, which cannot be true when the achieved level tracks the target
    /// cleanly across 10 dB). That question needs ears, not this harness.
    #[test]
    fn how_far_the_target_can_be_pushed() {
        {
            // The crest gate's margin, measured on both stimuli.
            let mut c = Normalizer::new(settings(true));
            let mut sp = Speech::new(-30.0);
            for _ in 0..400 { let mut b = sp.block(); c.process(&mut b, RATE); }
            let speech_crest = c.crest_db();
            let mut c2 = Normalizer::new(settings(true));
            let mut hum = Tone::new(120.0, -30.0);
            for _ in 0..400 { let mut b = hum.block(); c2.process(&mut b, RATE); }
            println!(
                "\n crest ramp {:.0}–{:.0} dB — speech {:.1} dB, steady tone {:.1} dB",
                CREST_NONE_DB, CREST_FULL_DB, speech_crest, c2.crest_db(),
            );
        }

        println!("\n target | achieved rms | gain");
        for target_db in [-20.0f32, -18.0, -16.0, -14.0, -12.0, -10.0] {
            let mut compressor = Normalizer::new(NormalizerSettings {
                enabled: true,
                target_rms: dbfs(target_db),
                ..Default::default()
            });
            // A quiet iPhone, the case the compressor exists for.
            let mut speech = Speech::new(-30.0);
            let mut sum_sq = 0.0f64;
            let mut samples = 0usize;
            for index in 0..600 {
                let mut block = speech.block();
                compressor.process(&mut block, RATE);
                if index >= 300 {
                    for sample in block.iter() {
                        sum_sq += (*sample as f64) * (*sample as f64);
                        samples += 1;
                    }
                }
            }
            let rms = (sum_sq / samples as f64).sqrt() as f32;
            println!(
                "{:>6.0} | {:>12.1} | {:>4.1}x",
                target_db,
                20.0 * (rms / FULL_SCALE).log10(),
                compressor.current_gain(),
            );
        }
        println!();
    }

    /// The emulator case, 2026-10-01: a steady hum is the ideal input for a level
    /// follower, and before the crest gate the compressor lifted it to the target
    /// and `#echo` played back a -20 dBFS drone.
    #[test]
    fn a_steady_hum_is_never_lifted() {
        let mut compressor = Normalizer::new(settings(true));
        let mut hum = Tone::new(120.0, -30.0);
        for _ in 0..600 {
            let mut block = hum.block();
            compressor.process(&mut block, RATE);
        }
        assert!(
            compressor.current_gain() <= 1.001,
            "a tone must not be amplified, gain reached {:.2}x (crest {:.1} dB)",
            compressor.current_gain(),
            compressor.crest_db(),
        );
    }

    /// And the gate must not block real speech, which is the other half.
    #[test]
    fn speech_passes_the_crest_gate() {
        let mut compressor = Normalizer::new(settings(true));
        let mut speech = Speech::new(-30.0);
        for _ in 0..400 {
            let mut block = speech.block();
            compressor.process(&mut block, RATE);
        }
        assert!(
            compressor.crest_db() > CREST_NONE_DB,
            "speech-shaped audio measured only {:.1} dB of crest, below the floor",
            compressor.crest_db(),
        );
        assert!(
            compressor.current_gain() > 1.2,
            "speech must earn some lift, got {:.2}x",
            compressor.current_gain(),
        );
    }

    /// What a downward compressor would actually BUY us, in decibels of
    /// loudness, over what the current chain already achieves.
    ///
    /// The honest comparison is not "before vs after compression" — our limiter
    /// already lets the level follower push the peak to the ceiling. The
    /// question is how much EXTRA average level a compressor buys by pulling the
    /// loudest syllables down and making room. So the baseline here is
    /// peak-normalising to the ceiling with no compression at all, which is what
    /// we ship today, and the payoff is the RMS above that.
    ///
    /// Printed with `--nocapture`; not an assertion.
    #[test]
    #[ignore = "15 s exploration, not a regression test: cargo test -- --ignored --nocapture"]
    fn what_a_downward_compressor_would_buy() {
        const CEILING: f32 = -1.0;

        fn peak_of(x: &[f32]) -> f32 { x.iter().fold(0.0f32, |a, s| a.max(s.abs())) }
        fn rms_of(x: &[f32]) -> f32 {
            (x.iter().map(|s| (*s as f64) * (*s as f64)).sum::<f64>() / x.len() as f64).sqrt() as f32
        }
        fn db(x: f32) -> f32 { 20.0 * (x / FULL_SCALE).log10() }

        /// Textbook downward compressor: peak envelope, hard knee, make-up left
        /// to the caller.
        fn compress(x: &[f32], threshold_db: f32, ratio: f32) -> Vec<f32> {
            // LOOK-AHEAD envelope. A feed-forward compressor builds its envelope
            // from what already went past, so the first sample of every transient
            // escapes at full gain: the peak survives, the make-up gain has
            // nothing to recover, and the body alone gets squashed — which RAISES
            // the crest factor. Measured it that way round twice before seeing it.
            // We can do better because the capture hook hands over a whole 10 ms
            // block before anything is written back, so the peak of what is coming
            // is knowable. That is the same free look-ahead the limiter already
            // uses.
            let look = (0.010 * RATE as f32) as usize; // 10 ms, one capture block
            let ahead: Vec<f32> = (0..x.len())
                .map(|i| x[i..(i + look).min(x.len())].iter().fold(0.0f32, |a, s| a.max(s.abs())))
                .collect();
            let release = (-1.0f32 / (0.080 * RATE as f32)).exp();  // 80 ms
            let threshold = dbfs(threshold_db);
            let mut env = 0.0f32;
            x.iter()
                .zip(ahead.iter())
                .map(|(sample, coming)| {
                    // Instant attack is safe with look-ahead; release stays slow.
                    env = if *coming > env { *coming } else { *coming + release * (env - *coming) };
                    let gain = if env > threshold && env > 0.0 {
                        let over_db = db(env) - threshold_db;
                        10f32.powf(-(over_db * (1.0 - 1.0 / ratio)) / 20.0)
                    } else {
                        1.0
                    };
                    sample * gain
                })
                .collect()
        }

        for (label, rms_dbfs) in [("Android-like", -18.0f32), ("iPhone-like", -30.0f32)] {
            let mut source = Speech::new(rms_dbfs);
            let mut signal = Vec::new();
            for _ in 0..400 { signal.extend(source.block()); }

            let raw_crest = db(peak_of(&signal)) - db(rms_of(&signal));
            // Baseline = today's chain: lift until the peak hits the ceiling.
            // The compressor is then measured ON TOP of that, which is also why
            // its threshold has to be relative to the ceiling rather than an
            // absolute level — a -30 dBFS input sits below every absolute
            // threshold and the compressor would simply never engage.
            let headroom = dbfs(CEILING) / peak_of(&signal);
            let normalised: Vec<f32> = signal.iter().map(|s| s * headroom).collect();
            let baseline_rms = db(rms_of(&normalised));

            println!(
                "\n {label}: input {:.1} dBFS rms, crest {:.1} dB → today's chain gives {:.1} dBFS",
                db(rms_of(&signal)), raw_crest, baseline_rms,
            );
            println!("   threshold | ratio | result rms | EXTRA over today | crest after");
            for threshold_db in [-12.0f32, -9.0, -6.0] {
                for ratio in [2.0f32, 3.0, 4.0] {
                    let compressed = compress(&normalised, threshold_db, ratio);
                    let makeup = dbfs(CEILING) / peak_of(&compressed).max(1e-9);
                    let out: Vec<f32> = compressed.iter().map(|s| s * makeup).collect();
                    println!(
                        "   {:>7.0} dB | {:>4.0}:1 | {:>9.1} | {:>+15.1} dB | {:>10.1} dB",
                        threshold_db, ratio,
                        db(rms_of(&out)),
                        db(rms_of(&out)) - baseline_rms,
                        db(peak_of(&out)) - db(rms_of(&out)),
                    );
                }
            }
        }
        println!();
    }

    #[test]
    fn disabled_is_bit_identical() {
        let mut compressor = Normalizer::new(settings(false));
        let mut tone = Tone::new(440.0, -30.0);
        let original = tone.block();
        let mut block = original.clone();
        compressor.process(&mut block, RATE);
        assert_eq!(original, block);
    }

    #[test]
    fn lifts_a_quiet_iphone_toward_the_target() {
        // -30 dBFS RMS is what two independent iPhones measured in #echo.
        let mut compressor = Normalizer::new(settings(true));
        let mut speech = Speech::new(-30.0);
        let before = compressor.current_gain();
        for _ in 0..400 {
            let mut block = speech.block();
            compressor.process(&mut block, RATE);
        }
        let gain = compressor.current_gain();
        assert!(
            gain > before * 1.3,
            "a -30 dBFS speech-shaped signal should be lifted, gain only reached {gain:.2}"
        );
    }

    #[test]
    fn leaves_a_healthy_android_almost_alone() {
        // -18 dBFS RMS is where Android already sits; 2 dB of lift at most.
        let mut compressor = Normalizer::new(settings(true));
        let mut speech = Speech::new(-18.0);
        for _ in 0..400 {
            let mut block = speech.block();
            compressor.process(&mut block, RATE);
        }
        let gain = compressor.current_gain();
        assert!(gain < 2.0, "an already healthy level needs little lift, got {gain:.2}x");
        assert!(gain >= 1.0, "must never attenuate, got {gain:.2}x");
    }

    #[test]
    fn never_crosses_the_ceiling() {
        // A signal whose peaks are already at -1.5 dBFS, like a hot Android.
        let mut compressor = Normalizer::new(settings(true));
        let mut tone = Tone::new(440.0, -4.5);
        for _ in 0..400 {
            let mut block = tone.block();
            compressor.process(&mut block, RATE);
            let peak = peak_dbfs(&block);
            assert!(peak <= -1.0 + 0.01, "peak {peak:.2} dBFS crossed the ceiling");
        }
    }

    #[test]
    fn a_transient_after_quiet_speech_does_not_clip() {
        // The classic failure: gain wound up during quiet talk, then a shout.
        let mut compressor = Normalizer::new(settings(true));
        let mut quiet = Speech::new(-34.0);
        for _ in 0..300 {
            let mut block = quiet.block();
            compressor.process(&mut block, RATE);
        }
        assert!(compressor.current_gain() > 1.3, "the gain must actually wind up first");
        let mut shout = Tone::new(440.0, -6.0);
        for _ in 0..100 {
            let mut block = shout.block();
            compressor.process(&mut block, RATE);
            let peak = peak_dbfs(&block);
            assert!(peak <= -1.0 + 0.01, "peak {peak:.2} dBFS clipped after a level jump");
        }
    }

    #[test]
    fn silence_does_not_wind_the_gain_up() {
        let mut compressor = Normalizer::new(settings(true));
        let mut speech = Speech::new(-24.0);
        for _ in 0..200 {
            let mut block = speech.block();
            compressor.process(&mut block, RATE);
        }
        let after_speech = compressor.current_gain();

        // Ten seconds of pauses must not ask for more gain.
        for _ in 0..1000 {
            let mut block = vec![0.0f32; BLOCK];
            compressor.process(&mut block, RATE);
            assert!(block.iter().all(|s| *s == 0.0), "silence must stay silent");
        }
        let after_silence = compressor.current_gain();
        assert!(
            after_silence <= after_speech + 0.01,
            "gain crept from {after_speech:.2} to {after_silence:.2} during silence"
        );
    }

    #[test]
    fn gain_moves_slowly_enough_not_to_pump() {
        let mut compressor = Normalizer::new(settings(true));
        let mut tone = Speech::new(-34.0);
        let mut previous = compressor.current_gain();
        for _ in 0..300 {
            let mut block = tone.block();
            let gain = compressor.process(&mut block, RATE);
            let step_db = 20.0 * (gain / previous).log10();
            assert!(
                step_db <= GAIN_UP_DB_PER_SEC * 0.01 + 0.001,
                "gain jumped {step_db:.3} dB in one 10 ms block"
            );
            previous = gain;
        }
    }

    /// Reported from the field 2026-10-01: switching the level mode ran the
    /// "adding" readout up to maximum with nobody speaking. `setNormalizer` calls
    /// `reset`, and on the first block after a reset the crest window is one
    /// block long, so room noise measured as speech-shaped and the ratchet kept
    /// whatever it earned.
    #[test]
    fn a_settings_change_cannot_launch_the_gain_on_room_noise() {
        let mut normalizer = Normalizer::new(settings(true));
        let mut speech = Speech::new(-30.0);
        for _ in 0..300 {
            let mut block = speech.block();
            normalizer.process(&mut block, RATE);
        }
        assert!(normalizer.current_gain() > 1.2, "setup: speech should have earned lift");

        // Exactly what the bridge does when the user picks another level.
        normalizer.set_settings(NormalizerSettings { target_rms: dbfs(-14.0), ..settings(true) });
        normalizer.reset();

        let mut room = Noise::new(-45.0);
        for _ in 0..600 {
            let mut block = room.block();
            normalizer.process(&mut block, RATE);
        }
        assert!(
            normalizer.current_gain() <= 1.001,
            "room noise after a settings change reached {:.2}x (crest {:.1} dB)",
            normalizer.current_gain(),
            normalizer.crest_db(),
        );
    }

    /// The other half: room noise is not a quiet voice, however spiky it reads.
    #[test]
    fn room_noise_is_not_a_quiet_voice() {
        let mut normalizer = Normalizer::new(settings(true));
        let mut room = Noise::new(-45.0);
        for _ in 0..1000 {
            let mut block = room.block();
            normalizer.process(&mut block, RATE);
        }
        assert!(
            normalizer.current_gain() <= 1.001,
            "an empty room earned {:.2}x",
            normalizer.current_gain()
        );
    }

    /// Reported from the field 2026-10-01: the readout stayed on the last
    /// non-zero value and would not fall back. It was faithful — `level` never
    /// updates below the noise floor, so nothing could lower `desired`, and the
    /// ratchet floored it at the gain already applied. Forever.
    #[test]
    fn the_gain_is_given_back_when_the_voice_stops() {
        let mut normalizer = Normalizer::new(settings(true));
        let mut speech = Speech::new(-30.0);
        for _ in 0..400 {
            let mut block = speech.block();
            normalizer.process(&mut block, RATE);
        }
        assert!(normalizer.current_gain() > 1.3, "setup: speech should have earned lift");

        // Three seconds of nothing: hold, then release.
        for _ in 0..300 {
            let mut block = vec![0.0f32; BLOCK];
            normalizer.process(&mut block, RATE);
        }
        assert!(
            normalizer.current_gain() <= 1.001,
            "gain stayed at {:.2}x three seconds after the voice stopped",
            normalizer.current_gain()
        );
    }

    /// And the release must not fire inside a sentence, which is what the
    /// ratchet was protecting in the first place.
    #[test]
    fn a_pause_between_words_keeps_the_gain() {
        let mut normalizer = Normalizer::new(settings(true));
        let mut speech = Speech::new(-30.0);
        for _ in 0..400 {
            let mut block = speech.block();
            normalizer.process(&mut block, RATE);
        }
        let mid_sentence = normalizer.current_gain();

        // 500 ms, longer than any pause between words.
        for _ in 0..50 {
            let mut block = vec![0.0f32; BLOCK];
            normalizer.process(&mut block, RATE);
        }
        assert!(
            (normalizer.current_gain() - mid_sentence).abs() < 0.01,
            "a 500 ms pause moved the gain from {mid_sentence:.2}x to {:.2}x",
            normalizer.current_gain(),
        );
    }

    #[test]
    fn respects_max_gain_on_near_silence() {
        let mut compressor = Normalizer::new(settings(true));
        let mut whisper = Speech::new(-60.0);
        for _ in 0..600 {
            let mut block = whisper.block();
            compressor.process(&mut block, RATE);
        }
        assert!(
            compressor.current_gain() <= settings(true).max_gain + 0.01,
            "gain {} exceeded max_gain",
            compressor.current_gain()
        );
    }
}
