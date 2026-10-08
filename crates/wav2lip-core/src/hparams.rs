use serde::{Deserialize, Serialize};

/// Canonical hyperparameters for Wav2Lip training and inference matching original hparams.py
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HParams {
    pub num_mels: usize,
    pub rescale: bool,
    pub rescaling_max: f32,
    pub use_lws: bool,
    pub n_fft: usize,
    pub hop_size: usize,
    pub win_size: usize,
    pub sample_rate: u32,
    pub fmin: f32,
    pub fmax: f32,
    pub img_size: usize,
    pub fps: f32,
    pub batch_size: usize,
    pub initial_learning_rate: f32,
    pub syncnet_batch_size: usize,
    pub syncnet_lr: f32,
    pub syncnet_eval_interval: usize,
    pub syncnet_checkpoint_interval: usize,
}

impl Default for HParams {
    fn default() -> Self {
        Self {
            num_mels: 80,
            rescale: true,
            rescaling_max: 0.9,
            use_lws: false,
            n_fft: 800,
            hop_size: 200,
            win_size: 800,
            sample_rate: 16000,
            fmin: 55.0,
            fmax: 7600.0,
            img_size: 96,
            fps: 25.0,
            batch_size: 16,
            initial_learning_rate: 1e-4,
            syncnet_batch_size: 64,
            syncnet_lr: 1e-4,
            syncnet_eval_interval: 10000,
            syncnet_checkpoint_interval: 10000,
        }
    }
}
