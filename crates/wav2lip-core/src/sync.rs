use serde::{Deserialize, Serialize};

/// Audio-visual sync evaluation score across temporal offsets [-5..+5 frames]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncEvaluation {
    pub best_offset_frames: i32,
    pub min_sync_distance: f32,
    pub confidence_score: f32,
    pub offset_distances: Vec<(i32, f32)>,
}

pub struct SyncNetEvaluator;

impl SyncNetEvaluator {
    /// Compute cosine distance between audio embedding and visual mouth embedding
    pub fn cosine_distance(audio_emb: &[f32], video_emb: &[f32]) -> f32 {
        if audio_emb.len() != video_emb.len() || audio_emb.is_empty() {
            return 1.0;
        }

        let mut dot = 0.0f32;
        let mut norm_a = 0.0f32;
        let mut norm_v = 0.0f32;

        for (&a, &v) in audio_emb.iter().zip(video_emb.iter()) {
            dot += a * v;
            norm_a += a * a;
            norm_v += v * v;
        }

        let denom = (norm_a.sqrt() * norm_v.sqrt()).max(1e-6);
        let similarity = (dot / denom).clamp(-1.0, 1.0);
        1.0 - similarity
    }

    /// Evaluate temporal synchronization alignment over window of offsets
    pub fn evaluate_alignment(
        audio_embeddings: &[Vec<f32>],
        video_embeddings: &[Vec<f32>],
        max_offset: i32,
    ) -> SyncEvaluation {
        let mut offset_distances = Vec::new();
        let total_frames = audio_embeddings.len().min(video_embeddings.len());

        let mut best_offset = 0;
        let mut min_dist = f32::MAX;

        for offset in -max_offset..=max_offset {
            let mut sum_dist = 0.0f32;
            let mut count = 0;

            for i in 0..total_frames {
                let v_idx = (i as i32 + offset) as usize;
                if v_idx < video_embeddings.len() {
                    let d = Self::cosine_distance(&audio_embeddings[i], &video_embeddings[v_idx]);
                    sum_dist += d;
                    count += 1;
                }
            }

            let avg_dist = if count > 0 {
                sum_dist / count as f32
            } else {
                1.0
            };
            offset_distances.push((offset, avg_dist));

            if avg_dist < min_dist {
                min_dist = avg_dist;
                best_offset = offset;
            }
        }

        // Confidence: difference between median distance and minimum distance
        let mut sorted_dists: Vec<f32> = offset_distances.iter().map(|(_, d)| *d).collect();
        sorted_dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median_dist = sorted_dists[sorted_dists.len() / 2];
        let confidence = (median_dist - min_dist).max(0.0) * 10.0;

        SyncEvaluation {
            best_offset_frames: best_offset,
            min_sync_distance: min_dist,
            confidence_score: confidence,
            offset_distances,
        }
    }
}
