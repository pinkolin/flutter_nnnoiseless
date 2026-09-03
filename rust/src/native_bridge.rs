use nnnoiseless::{DenoiseState, RnnModel};
use once_cell::sync::Lazy;
use std::sync::Mutex;

const TARGET_SAMPLE_RATE: u32 = 48_000;
const FRAME_SIZE: usize = DenoiseState::FRAME_SIZE;

static MODEL: Lazy<&'static RnnModel> =
    Lazy::new(|| Box::leak(Box::new(RnnModel::default())));

struct CaptureState {
    denoisers: Vec<Box<DenoiseState<'static>>>,
}

static CAPTURE_STATE: Lazy<Mutex<CaptureState>> =
    Lazy::new(|| Mutex::new(CaptureState { denoisers: Vec::new() }));

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

        state.denoisers[channel_index].process_frame(&mut output_frame, &input_frame);

        for (idx, sample) in samples[start..end].iter_mut().enumerate() {
            *sample = output_frame[idx].clamp(i16::MIN as f32, i16::MAX as f32);
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

#[cfg(target_os = "android")]
mod android {
    use super::{process_f32_channel_in_place, reset_capture_state};
    use jni::objects::{JClass, JFloatArray};
    use jni::sys::{jboolean, jint, JNI_FALSE, JNI_TRUE};
    use jni::JNIEnv;

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
