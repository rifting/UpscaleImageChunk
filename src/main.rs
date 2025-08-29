use clap::Parser; 
use image::{GenericImageView, RgbaImage};
use cust::prelude::*;
use std::error::Error;
use std::time::Instant;

/// Find the highest quality crop in an image
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Cropped image / Chunk from the original image
    #[arg(short, long)]
    cropped_image: String,

    /// Full image
    #[arg(short, long)]
    full_image: String,

    /// Output file
    #[arg(short, long, default_value = "out.png")]
    output: String,

    /// Minimum length of the cropped image (default = min side of cropped image).
    #[arg(short, long)]
    min_length: Option<u32>,

    /// Stop comparisons once the cropped chunk is this size 
    /// (default = maximum size before chunk touches full image edge).
    #[arg(short, long)]
    stop_at: Option<u32>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    let full = image::open(&args.full_image)?.to_rgba8();
    let chunk_orig = image::open(&args.cropped_image)?.to_rgba8();
    let (full_w, full_h) = full.dimensions();
    let (orig_w, orig_h) = chunk_orig.dimensions();
    let aspect_ratio = orig_w as f32 / orig_h as f32;

    // Defaults for min_length and stop_at if not provided
    let min_length = args.min_length.unwrap_or(orig_w.min(orig_h));

    let stop_at = args.stop_at.unwrap_or_else(|| {
        if aspect_ratio >= 1.0 {
            let max_w = full_w;
            let max_h = (max_w as f32 / aspect_ratio).round() as u32;
            if max_h > full_h {
                // limit by height
                full_h
            } else {
                // limit by width
                max_w
            }
        } else {
            let max_h = full_h;
            let max_w = (max_h as f32 * aspect_ratio).round() as u32;
            if max_w > full_w {
                // limit by width
                full_w
            } else {
                // limit by height
                max_h
            }
        }
    });

    println!("cropfit");
    println!();
    println!("Using min_length={}, stop_at={}", min_length, stop_at);

    let _ctx = cust::quick_init()?;
    let ptx = include_str!("ssd_kernel.ptx"); 
    let module = Module::from_ptx(ptx, &[])?;
    let function = module.get_function("ssd_kernel")?;

    let mut best_score = f64::MAX;
    let mut best_coords = (0, 0, 0, 0);

    let total_sizes = stop_at - min_length + 1;
    let start_time = Instant::now();

    let mut size = min_length;
    while size <= stop_at {
        let iter_start = Instant::now();

        let (chunk_w, chunk_h) = if aspect_ratio >= 1.0 {
            (size, (size as f32 / aspect_ratio).round() as u32)
        } else {
            ((size as f32 * aspect_ratio).round() as u32, size)
        };
        if chunk_w > full_w || chunk_h > full_h { break; }

        let chunk = image::imageops::resize(
            &chunk_orig,
            chunk_w,
            chunk_h,
            image::imageops::FilterType::Lanczos3,
        );

        let full_buf = full.as_raw();
        let chunk_buf = chunk.as_raw();
        let num_positions = (full_w - chunk_w + 1) * (full_h - chunk_h + 1);

        let full_gpu = DeviceBuffer::from_slice(&full_buf)?;
        let chunk_gpu = DeviceBuffer::from_slice(&chunk_buf)?;
        let mut scores_gpu = DeviceBuffer::<f64>::zeroed(num_positions as usize)?;

        let block_size = 256;
        let grid_size = (num_positions as u32 + block_size - 1) / block_size;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        unsafe {
            launch!(
                function<<<grid_size, block_size, 0, stream>>>(
                    full_gpu.as_device_ptr(),
                    chunk_gpu.as_device_ptr(),
                    full_w,
                    full_h,
                    chunk_w,
                    chunk_h,
                    scores_gpu.as_device_ptr()
                )
            )?;
        }

        let mut scores = vec![0f64; num_positions as usize];
        scores_gpu.copy_to(&mut scores)?;

        let (min_idx, &score) = scores.iter().enumerate().min_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap();
        if score < best_score {
            let x = (min_idx as u32) % (full_w - chunk_w + 1);
            let y = (min_idx as u32) / (full_w - chunk_w + 1);
            best_score = score;
            best_coords = (x, y, chunk_w, chunk_h);
            println!("New best normalized SSD {:.2} at x={}, y={}", score, x, y);
        }

        let iter_elapsed = iter_start.elapsed().as_secs_f64();
        let completed = size - min_length + 1;
        let remaining = total_sizes - completed;
        let est_remaining_time = remaining as f64 * iter_elapsed;

        println!(
            "Size {} processed in {:.2}s. Progress: {}/{} ({:.2}%). Estimated time left: {:.2}s",
            size,
            iter_elapsed,
            completed,
            total_sizes,
            (completed as f64 / total_sizes as f64) * 100.0,
            est_remaining_time
        );

        size += 1;
    }

    let (x, y, w, h) = best_coords;
    let best_region = full.view(x, y, w, h).to_image();
    best_region.save(&args.output)?;
    println!("Saved best chunk at x={}, y={}, size {}x{}", x, y, w, h);
    println!("Saved to {}", &args.output);
    println!("Total elapsed time: {:.2}s", start_time.elapsed().as_secs_f64());

    Ok(())
}
