use nnnoiseless::{DenoiseState, RnnModel};
use once_cell::sync::Lazy;
use std::sync::Mutex;

const TARGET_SAMPLE_RATE: u32 = 48_000;
const FRAME_SIZE: usize = DenoiseState::FRAME_SIZE;

static MODEL: Lazy<&'static RnnModel> =
    Lazy::new(|| Box::leak(Box::new(RnnModel::default())));

/// VOX gate driven by RNNoise's per-frame speech probability (see AUDIO_KB
/// "VOX with speech detection"). Channel 0 decides, other channels follow.
/// While closed, the denoised output is zeroed in place → Opus DTX, no
/// remote active-speaker blips, nobody hears the room.
struct VoxGate {
    enabled: bool,
    open: bool,
    open_thr: f32,
    close_thr: f32,
    attack_frames: u32,
    hangover_frames: u32,
    above: u32,
    below: u32,
    since_high: u32,
    last_prob: f32,
    transitions: u64,
}

impl VoxGate {
    const fn new() -> Self {
        VoxGate {
            enabled: false,
            open: false,
            open_thr: 0.5,
            close_thr: 0.2,
            attack_frames: 2,
            hangover_frames: 70,
            above: 0,
            below: 0,
            since_high: 0,
            last_prob: 0.0,
            transitions: 0,
        }
    }

    fn reset_runtime(&mut self) {
        self.open = false;
        self.above = 0;
        self.below = 0;
        self.since_high = 0;
        self.last_prob = 0.0;
    }

    /// Feed one frame's speech probability; returns whether the gate is open
    /// for this frame.
    fn update(&mut self, prob: f32) -> bool {
        self.last_prob = prob;
        if !self.enabled {
            return true;
        }
        if prob >= self.open_thr {
            self.above = self.above.saturating_add(1);
            self.below = 0;
            self.since_high = 0;
        } else {
            self.above = 0;
            self.since_high = self.since_high.saturating_add(1);
            if prob < self.close_thr {
                self.below = self.below.saturating_add(1);
            }
        }
        if !self.open && self.above >= self.attack_frames {
            self.open = true;
            self.transitions += 1;
        } else if self.open
            && (self.below >= self.hangover_frames
                || self.since_high >= self.hangover_frames.saturating_mul(2).max(1))
        {
            self.open = false;
            self.below = 0;
            self.transitions += 1;
        }
        self.open
    }
}

struct CaptureState {
    denoisers: Vec<Box<DenoiseState<'static>>>,
    gate: VoxGate,
}

static CAPTURE_STATE: Lazy<Mutex<CaptureState>> =
    Lazy::new(|| Mutex::new(CaptureState { denoisers: Vec::new(), gate: VoxGate::new() }));

/// Configure the VOX gate. Thresholds are RNNoise speech probabilities (0..1),
/// frame counts are 10 ms units. Disabling also closes/resets the runtime state.
pub fn set_vox_gate(enabled: bool, open_thr: f32, close_thr: f32, attack_frames: u32, hangover_frames: u32) {
    if let Ok(mut state) = CAPTURE_STATE.lock() {
        let gate = &mut state.gate;
        gate.enabled = enabled;
        gate.open_thr = open_thr.clamp(0.0, 1.0);
        gate.close_thr = close_thr.clamp(0.0, gate.open_thr);
        gate.attack_frames = attack_frames.max(1);
        gate.hangover_frames = hangover_frames.max(1);
        gate.reset_runtime();
    }
}

pub fn vox_gate_is_open() -> bool {
    CAPTURE_STATE.lock().map(|s| s.gate.enabled && s.gate.open).unwrap_or(false)
}

pub fn vox_gate_last_prob() -> f32 {
    CAPTURE_STATE.lock().map(|s| s.gate.last_prob).unwrap_or(0.0)
}

pub fn vox_gate_transitions() -> u64 {
    CAPTURE_STATE.lock().map(|s| s.gate.transitions).unwrap_or(0)
}

fn ensure_denoisers(state: &mut CaptureState, channels: usize) {
    if state.denoisers.len() == channels {
        return;
    }

    state.denoisers = (0..channels)
        .map(|_| DenoiseState::with_model(*MODEL))
        .collect();
}

pub fn reset_capture_state() {
    if let Ok(mut state) = CAPTURE_STATE.lock() {
        state.denoisers.clear();
        state.gate.reset_runtime();
    }
}

pub fn process_f32_channel_in_place(
    samples: &mut [f32],
    sample_rate: u32,
    channel_index: usize,
) -> bool {
    if samples.is_empty() {
        return true;
    }

    if sample_rate != TARGET_SAMPLE_RATE {
        return false;
    }

    if samples.len() < FRAME_SIZE {
        return true;
    }

    let blocks = samples.len() / FRAME_SIZE;
    let mut state = match CAPTURE_STATE.lock() {
        Ok(state) => state,
        Err(_) => return false,
    };

    ensure_denoisers(&mut state, channel_index + 1);

    for block in 0..blocks {
        let start = block * FRAME_SIZE;
        let end = start + FRAME_SIZE;

        let mut input_frame = vec![0.0f32; FRAME_SIZE];
        let mut output_frame = vec![0.0f32; FRAME_SIZE];

        for (idx, sample) in samples[start..end].iter().enumerate() {
            // On iOS/flutter_webrtc, RTCAudioBuffer rawBuffer(forChannel:) exposes
            // float samples in PCM16 amplitude space, not normalized [-1.0, 1.0].
            // Re-scaling by i16::MAX here corrupts the signal and effectively kills TX.
            input_frame[idx] = (*sample).clamp(i16::MIN as f32, i16::MAX as f32);
        }

        let prob = state.denoisers[channel_index].process_frame(&mut output_frame, &input_frame);

        // Channel 0 drives the VOX gate; other channels just follow its state.
        let open = if channel_index == 0 {
            state.gate.update(prob)
        } else {
            !state.gate.enabled || state.gate.open
        };

        if open {
            for (idx, sample) in samples[start..end].iter_mut().enumerate() {
                *sample = output_frame[idx].clamp(i16::MIN as f32, i16::MAX as f32);
            }
        } else {
            for sample in samples[start..end].iter_mut() {
                *sample = 0.0;
            }
        }
    }

    true
}

#[unsafe(no_mangle)]
pub extern "C" fn ketska_nnnoiseless_reset_capture_state() {
    reset_capture_state();
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn ketska_nnnoiseless_process_f32_channel(
    samples: *mut f32,
    frame_count: i32,
    sample_rate: i32,
    channel_index: i32,
) -> bool {
    if samples.is_null() || frame_count <= 0 || sample_rate <= 0 || channel_index < 0 {
        return false;
    }

    let slice = unsafe { std::slice::from_raw_parts_mut(samples, frame_count as usize) };
    process_f32_channel_in_place(
        slice,
        sample_rate as u32,
        channel_index as usize,
    )
}

#[unsafe(no_mangle)]
pub extern "C" fn ketska_nnnoiseless_set_vox_gate(
    enabled: bool,
    open_thr: f32,
    close_thr: f32,
    attack_frames: i32,
    hangover_frames: i32,
) {
    set_vox_gate(enabled, open_thr, close_thr, attack_frames.max(0) as u32, hangover_frames.max(0) as u32);
}

#[unsafe(no_mangle)]
pub extern "C" fn ketska_nnnoiseless_vox_gate_is_open() -> bool {
    vox_gate_is_open()
}

#[unsafe(no_mangle)]
pub extern "C" fn ketska_nnnoiseless_vox_gate_last_prob() -> f32 {
    vox_gate_last_prob()
}

#[unsafe(no_mangle)]
pub extern "C" fn ketska_nnnoiseless_vox_gate_transitions() -> u64 {
    vox_gate_transitions()
}

#[cfg(target_os = "android")]
mod android {
    use super::{
        process_f32_channel_in_place, reset_capture_state, set_vox_gate, vox_gate_is_open,
        vox_gate_last_prob, vox_gate_transitions,
    };
    use jni::objects::{JClass, JFloatArray};
    use jni::sys::{jboolean, jfloat, jint, jlong, JNI_FALSE, JNI_TRUE};
    use jni::JNIEnv;

    #[unsafe(no_mangle)]
    pub extern "system" fn Java_cz_ketska_ketska_1app_KetskaAiFilterNative_nativeSetVoxGate(
        _env: JNIEnv,
        _class: JClass,
        enabled: jboolean,
        open_thr: jfloat,
        close_thr: jfloat,
        attack_frames: jint,
        hangover_frames: jint,
    ) {
        set_vox_gate(
            enabled != JNI_FALSE,
            open_thr,
            close_thr,
            attack_frames.max(0) as u32,
            hangover_frames.max(0) as u32,
        );
    }

    #[unsafe(no_mangle)]
    pub extern "system" fn Java_cz_ketska_ketska_1app_KetskaAiFilterNative_nativeVoxGateIsOpen(
        _env: JNIEnv,
        _class: JClass,
    ) -> jboolean {
        if vox_gate_is_open() { JNI_TRUE } else { JNI_FALSE }
    }

    #[unsafe(no_mangle)]
    pub extern "system" fn Java_cz_ketska_ketska_1app_KetskaAiFilterNative_nativeVoxGateLastProb(
        _env: JNIEnv,
        _class: JClass,
    ) -> jfloat {
        vox_gate_last_prob()
    }

    #[unsafe(no_mangle)]
    pub extern "system" fn Java_cz_ketska_ketska_1app_KetskaAiFilterNative_nativeVoxGateTransitions(
        _env: JNIEnv,
        _class: JClass,
    ) -> jlong {
        vox_gate_transitions() as jlong
    }

    #[unsafe(no_mangle)]
    pub extern "system" fn Java_cz_ketska_ketska_1app_KetskaAiFilterNative_nativeResetCaptureState(
        _env: JNIEnv,
        _class: JClass,
    ) {
        reset_capture_state();
    }

    /// WebRTC's Android capture post-processing hook (webrtc-sdk
    /// `ExternalAudioProcessor::Process`) hands the app `audio->channels()[0]`:
    /// the FULL-BAND mono signal as float32 in PCM16 amplitude space (±32768),
    /// `num_frames` samples per 10 ms. `num_bands` is informational only (it
    /// sizes the buffer: 160 × bands = frames); the data is never band-split.
    /// One 10 ms block at 48 kHz is exactly one RNNoise frame (480 samples).
    #[unsafe(no_mangle)]
    pub extern "system" fn Java_cz_ketska_ketska_1app_KetskaAiFilterNative_nativeProcessCaptureFloatBuffer(
        env: JNIEnv,
        _class: JClass,
        samples: JFloatArray,
        frame_count: jint,
        sample_rate: jint,
    ) -> jboolean {
        if frame_count <= 0 || sample_rate <= 0 {
            return JNI_FALSE;
        }

        let len = match env.get_array_length(&samples) {
            Ok(value) => value as usize,
            Err(_) => return JNI_FALSE,
        };
        let frames = (frame_count as usize).min(len);
        if frames == 0 {
            return JNI_FALSE;
        }

        let mut buffer = vec![0.0f32; frames];
        if env.get_float_array_region(&samples, 0, &mut buffer).is_err() {
            return JNI_FALSE;
        }

        let processed = process_f32_channel_in_place(&mut buffer, sample_rate as u32, 0);

        if processed {
            if env.set_float_array_region(&samples, 0, &buffer).is_err() {
                return JNI_FALSE;
            }
            JNI_TRUE
        } else {
            JNI_FALSE
        }
    }
}
