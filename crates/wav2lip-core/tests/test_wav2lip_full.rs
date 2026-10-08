use wav2lip_core::{
    FaceCropPatch, FaceMasker, HParams, SyncNetEvaluator,
    TensorShape4D, Wav2LipAudio, Wav2LipGenerator, WeightsManager,
    WAV2LIP_HOP_SIZE, WAV2LIP_MEL_BANDS, WAV2LIP_MEL_STEP_SIZE, WAV2LIP_SAMPLE_RATE,
};

#[test]
fn test_hparams_specification() {
    let hp = HParams::default();
    assert_eq!(hp.sample_rate, 16000);
    assert_eq!(hp.n_fft, 800);
    assert_eq!(hp.hop_size, 200);
    assert_eq!(hp.num_mels, 80);
    assert_eq!(hp.img_size, 96);
    assert_eq!(hp.fps, 25.0);
    assert_eq!(hp.fmin, 55.0);
    assert_eq!(hp.fmax, 7600.0);
}

#[test]
fn test_audio_constants_and_resampling() {
    assert_eq!(WAV2LIP_SAMPLE_RATE, 16000);
    assert_eq!(WAV2LIP_MEL_BANDS, 80);
    assert_eq!(WAV2LIP_HOP_SIZE, 200);
    assert_eq!(WAV2LIP_MEL_STEP_SIZE, 16);

    // Resample 48000Hz sine to 16000Hz
    let src_len = 48000;
    let input: Vec<f32> = (0..src_len).map(|i| (i as f32 * 0.1).sin()).collect();
    let resampled = Wav2LipAudio::resample(&input, 48000, 16000);
    assert_eq!(resampled.len(), 16000);
}

#[test]
fn test_face_lower_half_mask_and_patch() {
    let mask = FaceMasker::create_lower_half_mask(96);
    assert_eq!(mask.len(), 96 * 96);

    // Top rows should have 0 alpha (unmasked)
    for y in 0..40 {
        assert_eq!(mask[y * 96 + 48], 0.0);
    }
    // Bottom rows should have 1 alpha (masked)
    for y in 52..96 {
        assert_eq!(mask[y * 96 + 48], 1.0);
    }

    // Apply mask to RGB buffer (96 x 96 x 3)
    let mut rgb = vec![255u8; 96 * 96 * 3];
    FaceMasker::apply_mask_to_rgb(&mut rgb, 96, 96);

    // Top half remains untouched
    assert_eq!(rgb[10 * 96 * 3], 255);
    // Bottom half zeroed
    assert_eq!(rgb[80 * 96 * 3], 0);

    let patch = FaceCropPatch {
        width: 96,
        height: 96,
        bbox: [100, 150, 196, 246],
    };
    assert_eq!(patch.width, 96);
    assert_eq!(patch.height, 96);
}

#[test]
fn test_syncnet_evaluation_metrics() {
    // Perfect match
    let v1 = vec![0.5f32; 512];
    let v2 = vec![0.5f32; 512];
    let dist = SyncNetEvaluator::cosine_distance(&v1, &v2);
    assert!(dist < 1e-4);

    // Orthogonal
    let mut v_orth1 = vec![0.0f32; 512];
    let mut v_orth2 = vec![0.0f32; 512];
    v_orth1[0] = 1.0;
    v_orth2[1] = 1.0;
    let dist_orth = SyncNetEvaluator::cosine_distance(&v_orth1, &v_orth2);
    assert!((dist_orth - 1.0).abs() < 1e-4);

    // Alignment test across 10 frames
    let audio_feats = vec![vec![1.0f32; 512]; 10];
    let mut video_feats = vec![vec![0.0f32; 512]; 10];
    // Frame 5 has perfect match with frame 5
    video_feats[5] = vec![1.0f32; 512];

    let eval = SyncNetEvaluator::evaluate_alignment(&audio_feats, &video_feats, 3);
    assert_eq!(eval.offset_distances.len(), 7); // -3..=3
}

#[test]
fn test_wav2lip_generator_validation() {
    let gen = Wav2LipGenerator::new();
    assert_eq!(gen.latent_dim, 512);

    let valid_audio = TensorShape4D::new(1, 1, 80, 16);
    let valid_face = TensorShape4D::new(1, 6, 96, 96);
    assert!(gen.validate_inputs(valid_audio, valid_face).is_ok());

    let invalid_audio = TensorShape4D::new(1, 2, 80, 16);
    assert!(gen.validate_inputs(invalid_audio, valid_face).is_err());

    let invalid_face = TensorShape4D::new(1, 6, 128, 128);
    assert!(gen.validate_inputs(valid_audio, invalid_face).is_err());
}

#[test]
fn test_weights_manager() {
    let res = WeightsManager::inspect_checkpoint("/non/existent/path.safetensors");
    assert!(res.is_err());
}
