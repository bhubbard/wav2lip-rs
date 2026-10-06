use anyhow::{Context, Result};
use hound::WavReader;
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;
use std::path::Path;

pub const WAV2LIP_SAMPLE_RATE: u32 = 16000;
pub const WAV2LIP_MEL_BANDS: usize = 80;
pub const WAV2LIP_HOP_SIZE: usize = 200; // 12.5ms hop
pub const WAV2LIP_MEL_STEP_SIZE: usize = 16; // 16 time steps (0.2s context window per video frame)

/// 80-channel Mel chunk (80 x 16) centered on each video frame
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Wav2LipMelFrame {
    pub frame_idx: usize,
    pub mel_data: Vec<f32>, // 80 * 16 floats
}

pub struct Wav2LipAudio;

impl Wav2LipAudio {
    pub fn load_wav<P: AsRef<Path>>(path: P) -> Result<Vec<f32>> {
        let mut reader = WavReader::open(path.as_ref())
            .with_context(|| format!("Failed to read WAV file {:?}", path.as_ref()))?;
        
        let spec = reader.spec();
        let channels = spec.channels as usize;
        let source_rate = spec.sample_rate;

        let raw: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Int => {
                let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
                reader.samples::<i32>().map(|s| s.unwrap_or(0) as f32 / max).collect()
            }
            hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap_or(0.0)).collect(),
        };

        let mono = if channels > 1 {
            raw.chunks(channels).map(|c| c.iter().sum::<f32>() / channels as f32).collect()
        } else {
            raw
        };

        if source_rate != WAV2LIP_SAMPLE_RATE {
            Ok(Self::resample(&mono, source_rate, WAV2LIP_SAMPLE_RATE))
        } else {
            Ok(mono)
        }
    }

    pub fn resample(input: &[f32], src_rate: u32, target_rate: u32) -> Vec<f32> {
        if src_rate == target_rate || input.is_empty() {
            return input.to_vec();
        }
        let ratio = target_rate as f64 / src_rate as f64;
        let out_len = (input.len() as f64 * ratio).round() as usize;
        let mut out = Vec::with_capacity(out_len);
        for i in 0..out_len {
            let src_idx = i as f64 / ratio;
            let low = src_idx.floor() as usize;
            let high = (low + 1).min(input.len().saturating_sub(1));
            let frac = (src_idx - low as f64) as f32;
            out.push(input[low] * (1.0 - frac) + input[high] * frac);
        }
        out
    }

    /// Extract sequence of 16-step Mel spectrogram chunks matching video frame rate
    pub fn extract_mel_frames(samples: &[f32], fps: f32) -> Vec<Wav2LipMelFrame> {
        let samples_per_frame = (WAV2LIP_SAMPLE_RATE as f32 / fps).round() as usize;
        let total_samples = samples.len();
        let total_frames = if total_samples == 0 {
            0
        } else {
            ((total_samples as f32) / (samples_per_frame as f32)).ceil() as usize
        };

        // Window size for 16 mel steps = 16 * 200 hop = 3200 samples (0.2 seconds)
        let mel_window_samples = WAV2LIP_MEL_STEP_SIZE * WAV2LIP_HOP_SIZE;
        let mut frames = Vec::with_capacity(total_frames);

        for f in 0..total_frames {
            let center_sample = f * samples_per_frame + (samples_per_frame / 2);
            let start = center_sample.saturating_sub(mel_window_samples / 2);
            let end = (start + mel_window_samples).min(total_samples);

            let chunk = if start < total_samples {
                &samples[start..end]
            } else {
                &[]
            };

            let mel = Self::compute_mel_bands(chunk, WAV2LIP_MEL_BANDS, WAV2LIP_MEL_STEP_SIZE);
            frames.push(Wav2LipMelFrame {
                frame_idx: f,
                mel_data: mel,
            });
        }

        frames
    }

    fn compute_mel_bands(chunk: &[f32], bands: usize, steps: usize) -> Vec<f32> {
        let mut mel = vec![0.0; bands * steps];
        if chunk.is_empty() {
            return mel;
        }

        let step_size = (chunk.len() / steps).max(1);
        for t in 0..steps {
            let start = (t * step_size).min(chunk.len());
            let end = ((t + 1) * step_size).min(chunk.len());
            let slice = &chunk[start..end];
            if !slice.is_empty() {
                let sub_size = (slice.len() / bands).max(1);
                for b in 0..bands {
                    let b_start = (b * sub_size).min(slice.len());
                    let b_end = ((b + 1) * sub_size).min(slice.len());
                    let b_slice = &slice[b_start..b_end];
                    if !b_slice.is_empty() {
                        let mut energy = 0.0;
                        for (i, &s) in b_slice.iter().enumerate() {
                            let w = 0.54 - 0.46 * (2.0 * PI * i as f32 / b_slice.len() as f32).cos();
                            energy += (s * w).powi(2);
                        }
                        let log_energy = (1.0 + (energy / b_slice.len() as f32)).ln();
                        mel[t * bands + b] = log_energy;
                    }
                }
            }
        }

        mel
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_mel_frames() {
        let samples = vec![0.0; 16000]; // 1 second
        let frames = Wav2LipAudio::extract_mel_frames(&samples, 25.0);
        assert_eq!(frames.len(), 25);
        assert_eq!(frames[0].mel_data.len(), 80 * 16);
    }
}
