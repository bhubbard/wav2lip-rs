use serde::{Deserialize, Serialize};

/// 96x96 Cropped Face Representation with lower-half masking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaceCropPatch {
    pub width: u32,
    pub height: u32,
    pub bbox: [u32; 4], // [ymin, xmin, ymax, xmax]
}

pub struct FaceMasker;

impl FaceMasker {
    /// Create the canonical Wav2Lip lower-half face mask (96x96)
    /// Upper half (rows 0..48) retains identity (alpha = 0.0, unmasked)
    /// Lower half (rows 48..96) is masked for speech generation (alpha = 1.0, masked)
    pub fn create_lower_half_mask(size: usize) -> Vec<f32> {
        let mut mask = vec![0.0f32; size * size];
        let mid = size / 2;
        let feather = 6.0f32;

        for y in 0..size {
            let alpha = if y < mid.saturating_sub(3) {
                0.0
            } else if y <= mid + 3 {
                (y as f32 - (mid as f32 - 3.0)) / feather
            } else {
                1.0
            };

            for x in 0..size {
                mask[y * size + x] = alpha;
            }
        }

        mask
    }

    /// Mask bottom half of 96x96 RGB image buffer (3 channels, 96*96*3)
    pub fn apply_mask_to_rgb(rgb_data: &mut [u8], width: usize, height: usize) {
        let mid = height / 2;
        for y in mid..height {
            for x in 0..width {
                let idx = (y * width + x) * 3;
                if idx + 2 < rgb_data.len() {
                    rgb_data[idx] = 0;
                    rgb_data[idx + 1] = 0;
                    rgb_data[idx + 2] = 0;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lower_half_mask() {
        let mask = FaceMasker::create_lower_half_mask(96);
        assert_eq!(mask.len(), 96 * 96);
        assert_eq!(mask[10 * 96 + 48], 0.0); // Top half unmasked
        assert_eq!(mask[80 * 96 + 48], 1.0); // Bottom half masked
    }
}
