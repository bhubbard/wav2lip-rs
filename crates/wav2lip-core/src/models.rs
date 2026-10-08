use serde::{Deserialize, Serialize};

/// Layer specification for convolutional blocks in Wav2Lip and SyncNet
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvBlockSpec {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
    pub residual: bool,
}

/// Tensor shape helper: [Batch, Channels, Height, Width]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorShape4D {
    pub batch: usize,
    pub channels: usize,
    pub height: usize,
    pub width: usize,
}

impl TensorShape4D {
    pub fn new(batch: usize, channels: usize, height: usize, width: usize) -> Self {
        Self { batch, channels, height, width }
    }

    pub fn total_elements(&self) -> usize {
        self.batch * self.channels * self.height * self.width
    }
}

/// Wav2Lip Audio Encoder architecture specification:
/// Processes (B, 1, 80, 16) Mel chunk -> (B, 512, 1, 1) audio embedding
#[derive(Debug, Clone)]
pub struct AudioEncoder {
    pub blocks: Vec<ConvBlockSpec>,
}

impl AudioEncoder {
    pub fn new() -> Self {
        Self {
            blocks: vec![
                ConvBlockSpec { in_channels: 1, out_channels: 32, kernel_size: 3, stride: 1, padding: 1, residual: false },
                ConvBlockSpec { in_channels: 32, out_channels: 64, kernel_size: 3, stride: (2, 1).0, padding: 1, residual: false },
                ConvBlockSpec { in_channels: 64, out_channels: 128, kernel_size: 3, stride: (2, 2).0, padding: 1, residual: true },
                ConvBlockSpec { in_channels: 128, out_channels: 256, kernel_size: 3, stride: (2, 2).0, padding: 1, residual: true },
                ConvBlockSpec { in_channels: 256, out_channels: 512, kernel_size: 3, stride: (2, 2).0, padding: 1, residual: true },
                ConvBlockSpec { in_channels: 512, out_channels: 512, kernel_size: 3, stride: 1, padding: 0, residual: false },
            ],
        }
    }

    /// Calculate output shape from input shape
    pub fn output_shape(&self, input: TensorShape4D) -> TensorShape4D {
        let mut shape = input;
        for block in &self.blocks {
            shape.channels = block.out_channels;
            shape.height = (shape.height + 2 * block.padding - block.kernel_size) / block.stride + 1;
            shape.width = (shape.width + 2 * block.padding - block.kernel_size) / block.stride + 1;
        }
        shape
    }
}

/// Wav2Lip Face/Video Encoder architecture specification:
/// Processes 6-channel masked + target frame (B, 6, 96, 96) -> downsamples to (B, 512, 3, 3)
#[derive(Debug, Clone)]
pub struct FaceEncoder {
    pub blocks: Vec<ConvBlockSpec>,
}

impl FaceEncoder {
    pub fn new() -> Self {
        Self {
            blocks: vec![
                ConvBlockSpec { in_channels: 6, out_channels: 32, kernel_size: 7, stride: 1, padding: 3, residual: false },
                ConvBlockSpec { in_channels: 32, out_channels: 64, kernel_size: 3, stride: 2, padding: 1, residual: true },
                ConvBlockSpec { in_channels: 64, out_channels: 128, kernel_size: 3, stride: 2, padding: 1, residual: true },
                ConvBlockSpec { in_channels: 128, out_channels: 256, kernel_size: 3, stride: 2, padding: 1, residual: true },
                ConvBlockSpec { in_channels: 256, out_channels: 512, kernel_size: 3, stride: 2, padding: 1, residual: true },
                ConvBlockSpec { in_channels: 512, out_channels: 512, kernel_size: 3, stride: 2, padding: 1, residual: true },
            ],
        }
    }
}

/// Full Wav2Lip Generator coordinating Audio Encoder, Face Encoder, and Face Decoder
#[derive(Debug, Clone)]
pub struct Wav2LipGenerator {
    pub audio_encoder: AudioEncoder,
    pub face_encoder: FaceEncoder,
    pub latent_dim: usize,
}

impl Wav2LipGenerator {
    pub fn new() -> Self {
        Self {
            audio_encoder: AudioEncoder::new(),
            face_encoder: FaceEncoder::new(),
            latent_dim: 512,
        }
    }

    /// Validates tensor compatibility for inference
    pub fn validate_inputs(&self, audio_shape: TensorShape4D, face_shape: TensorShape4D) -> Result<(), String> {
        if audio_shape.batch != face_shape.batch {
            return Err(format!("Batch mismatch: audio {}, face {}", audio_shape.batch, face_shape.batch));
        }
        if audio_shape.channels != 1 || audio_shape.height != 80 || audio_shape.width != 16 {
            return Err(format!("Audio input must be [B, 1, 80, 16], got {:?}", audio_shape));
        }
        if face_shape.channels != 6 || face_shape.height != 96 || face_shape.width != 96 {
            return Err(format!("Face input must be [B, 6, 96, 96], got {:?}", face_shape));
        }
        Ok(())
    }
}

/// SyncNet Color Model architecture specification for lip sync confidence
#[derive(Debug, Clone)]
pub struct SyncNetModel {
    pub embedding_dim: usize,
}

impl SyncNetModel {
    pub fn new() -> Self {
        Self { embedding_dim: 512 }
    }

    /// Calculates L2 normalized cosine distance between 512-dim audio and video vectors
    pub fn score_similarity(&self, audio_feat: &[f32], video_feat: &[f32]) -> f32 {
        crate::sync::SyncNetEvaluator::cosine_distance(audio_feat, video_feat)
    }
}
