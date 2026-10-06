use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Wav2LipConfig {
    /// Canonical cropped face patch resolution (standard: 96x96)
    pub crop_size: u32,
    /// Bounding box padding [top, bottom, left, right]
    pub pads: [i32; 4],
    /// Video processing frame rate (default 25.0)
    pub fps: f32,
    /// Mel spectrogram time steps per frame (standard: 16)
    pub mel_step_size: usize,
    /// Disable temporal smoothing of bounding box
    pub nosmooth: bool,
    /// Treat source as a single static image portrait
    pub static_face: bool,
    /// Model checkpoint path
    pub checkpoint_path: Option<PathBuf>,
}

impl Default for Wav2LipConfig {
    fn default() -> Self {
        Self {
            crop_size: 96,
            pads: [0, 10, 0, 0],
            fps: 25.0,
            mel_step_size: 16,
            nosmooth: false,
            static_face: false,
            checkpoint_path: None,
        }
    }
}
