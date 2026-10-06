use anyhow::Result;
use clap::{Parser, Subcommand};
use colored::*;
use std::path::PathBuf;
use std::time::Instant;
use wav2lip_core::{Wav2LipAudio, Wav2LipConfig, Wav2LipPipeline, WeightsManager};

#[derive(Parser, Debug)]
#[command(
    name = "wav2lip",
    author = "Wav2Lip Contributors",
    version = "0.1.0",
    about = "High-performance Rust CLI for Wav2Lip: Accurately Lip-syncing Videos In The Wild"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Synchronize mouth movements in face video or image to match driving audio
    Sync {
        /// Source video or image file
        #[arg(short = 'f', long = "face", required = true)]
        face: PathBuf,

        /// Driving audio WAV file
        #[arg(short = 'a', long = "audio", required = true)]
        audio: PathBuf,

        /// Output path for synchronized video
        #[arg(short = 'o', long = "output", default_value = "results/result_voice.mp4")]
        output: PathBuf,

        /// Face bounding box padding [top, bottom, left, right]
        #[arg(long = "pads", num_args = 4, default_values_t = [0, 10, 0, 0])]
        pads: Vec<i32>,

        /// Prevent temporal smoothing of face bounding boxes
        #[arg(long = "nosmooth")]
        nosmooth: bool,

        /// Treat face input as a static still image
        #[arg(long = "static")]
        static_face: bool,
    },

    /// Inspect audio tracks or SafeTensors checkpoints
    Inspect {
        /// Target file
        #[arg(required = true)]
        file: PathBuf,
    },

    /// Benchmark 80-channel Mel spectrogram chunking and face masking
    Bench {
        /// Optional audio WAV file
        #[arg(short = 'a', long = "audio")]
        audio: Option<PathBuf>,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Sync {
            face,
            audio,
            output,
            pads,
            nosmooth,
            static_face,
        } => {
            println!("{}", "=== Wav2Lip Native Rust Lip-Sync Pipeline ===".bold().cyan());
            println!("  Input Face:  {}", face.display().to_string().yellow());
            println!("  Input Audio: {}", audio.display().to_string().yellow());
            println!("  Paddings:    {:?}", pads);
            println!("  Static Mode: {}", static_face);

            let config = Wav2LipConfig {
                crop_size: 96,
                pads: [pads[0], pads[1], pads[2], pads[3]],
                fps: 25.0,
                mel_step_size: 16,
                nosmooth,
                static_face,
                checkpoint_path: None,
            };

            let start = Instant::now();
            let pipeline = Wav2LipPipeline::new(config);
            let out_file = pipeline.run(&face, &audio, &output)?;

            println!(
                "{} Lip synchronization complete in {:.2?} -> {}",
                "✓".green(),
                start.elapsed(),
                out_file.display().to_string().bold()
            );
        }

        Commands::Inspect { file } => {
            println!("{} {}", "Inspecting:".bold().cyan(), file.display());
            let ext = file.extension().and_then(|s| s.to_str()).unwrap_or("");

            match ext.to_lowercase().as_str() {
                "wav" => {
                    let samples = Wav2LipAudio::load_wav(&file)?;
                    let frames = Wav2LipAudio::extract_mel_frames(&samples, 25.0);
                    println!("  Type:          WAV Audio Track");
                    println!("  Sample Rate:   16000 Hz");
                    println!("  Total Samples: {}", samples.len());
                    println!("  Mel Windows:   {} (16 time steps x 80 bands)", frames.len());
                }
                "safetensors" => {
                    let tensors = WeightsManager::inspect_checkpoint(&file)?;
                    println!("  Type:          SafeTensors Checkpoint");
                    println!("  Tensors Count: {}", tensors.len());
                    for t in tensors.iter().take(10) {
                        println!("    • {:<35} {:?} ({})", t.name, t.shape, t.dtype);
                    }
                    if tensors.len() > 10 {
                        println!("    ... and {} more tensors", tensors.len() - 10);
                    }
                }
                _ => {
                    println!("File format .{} not supported for detailed inspection", ext);
                }
            }
        }

        Commands::Bench { audio } => {
            println!("{}", "=== Benchmarking Wav2Lip Audio & Masking Engine ===".bold().cyan());
            let samples = if let Some(p) = audio {
                Wav2LipAudio::load_wav(&p)?
            } else {
                (0..160_000).map(|i| (i as f32 * 0.05).sin() * 0.35).collect()
            };

            let duration = samples.len() as f32 / 16000.0;
            println!("Testing on {:.1} seconds of 16kHz audio...", duration);

            let t0 = Instant::now();
            let mel_frames = Wav2LipAudio::extract_mel_frames(&samples, 25.0);
            let mel_dur = t0.elapsed();

            let t1 = Instant::now();
            let iterations = 1000;
            for _ in 0..iterations {
                let _ = wav2lip_core::FaceMasker::create_lower_half_mask(96);
            }
            let mask_dur = t1.elapsed() / iterations;

            println!("\nResults:");
            println!("  Mel Windows (80x16): {:.2?} ({} frames, {:.1} fps throughput)",
                mel_dur, mel_frames.len(), mel_frames.len() as f64 / mel_dur.as_secs_f64());
            println!("  Lower-Half Mask Gen: {:.2?} per 96x96 patch ({:.1} patches/sec)",
                mask_dur, 1.0 / mask_dur.as_secs_f64());
        }
    }

    Ok(())
}
