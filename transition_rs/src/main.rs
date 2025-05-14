// src/main.rs

mod args;
mod macros;
mod image_processing;
mod transitions;

use image_processing::ProcessedImage;
use opencv::{
    prelude::*,
    core::Size,
    highgui,
    imgcodecs::{imread, IMREAD_COLOR},
    Result,
};
use std::path::Path;
use clap::Parser;
use args::Args;
use log::{info, error};
use rand::Rng;

fn process_images(image_paths: Vec<&str>, base_path: &Path, size: i32, display_image: bool) -> Result<()> {
    // check if image exists
    for file_name in &image_paths {
        let path = base_path.with_file_name(file_name);
        if !path.exists() {
            panic!("Image file not found: {:?}", path);
        }
    }

    // Process images
    let mut images = Vec::new();
    for file_name in image_paths {
        let path = base_path.with_file_name(file_name);
        let image = opencv::imgcodecs::imread(&path.to_str().unwrap(), opencv::imgcodecs::IMREAD_COLOR)?;
        let mut processed_image = ProcessedImage::new(image, 8, true);
        let size = Some(Size::new(size, size)); // Example size
        let location = Some(vec![(100, 100), (200, 200)]); // Example locations
        processed_image.process(size, location)?;
        images.push(processed_image);
    }

    // Display images
    for (index, processed_image) in images.iter().enumerate() {
        let image = processed_image.image();
        let rgb_image = if processed_image.convert_to_lab {
            // convert back to RGB for display
            let mut rgb_image = Mat::default();
            opencv::imgproc::cvt_color(&image, &mut rgb_image, opencv::imgproc::COLOR_Lab2BGR, 0)?;
            rgb_image
        } else {
            image.clone()
        };

        let window_name = format!("Processed Image {}", index + 1);
        highgui::named_window(&window_name, highgui::WINDOW_AUTOSIZE)?;
        highgui::imshow(&window_name, &rgb_image)?;

        let window_name = format!("Flow Image {}", index + 1);
        highgui::named_window(&window_name, highgui::WINDOW_AUTOSIZE)?;
        highgui::imshow(&window_name, &processed_image.flow_image())?;
    }

    // Wait for a key press indefinitely
    highgui::wait_key(0)?;

    // Optionally, destroy all windows after key press
    highgui::destroy_all_windows()?;

    Ok(())
}

fn parse_start_location(start: Option<String>, image_size: (i32, i32)) -> Option<Vec<(i32, i32)>> {
    let location = if let Some(start_str) = start {
        // Parse the x,y coordinate string
        let coords: Vec<&str> = start_str.split(',').collect();
        if coords.len() == 2 {
            if let (Ok(x), Ok(y)) = (coords[0].parse::<i32>(), coords[1].parse::<i32>()) {
                // Clamp coordinates to image bounds
                let x = x.clamp(0, image_size.0 - 1);
                let y = y.clamp(0, image_size.1 - 1);
                info!("Using specified start location: ({}, {})", x, y);
                Some(vec![(x, y)])
            } else {
                error!("Invalid coordinate format. Using random location.");
                None
            }
        } else {
            error!("Invalid start location format. Expected 'x,y'. Using random location.");
            None
        }
    } else {
        None
    };

    Some(location.unwrap_or_else(|| {
        let mut rng = rand::thread_rng();
        let x = rng.gen_range(0..image_size.0);
        let y = rng.gen_range(0..image_size.1);
        info!("Using random start location: ({}, {})", x, y);
        vec![(x, y)]
    }))
}

fn main() -> opencv::Result<()> {
    // let image_paths = vec!["456_generated.jpg"];
    // let base_path = Path::new("../images/mock/gen/test.jpg");
    // let size = 256;
    // let _result = process_images(image_paths, base_path, size);

    // Parse command-line arguments
    let args = Args::parse();

    env_logger::init();
    
    // check that image exists
    let path = Path::new(&args.input);
    if !path.exists() {
        error!("Image file not found: {:?}", path);
        std::process::exit(1);
    }

    // Read image
    let image = imread(&args.input, IMREAD_COLOR)?;
    
    // Get image dimensions for location validation
    let image_size = (image.cols(), image.rows());
    info!("Image dimensions: {}x{}", image_size.0, image_size.1);
    let start_location = parse_start_location(args.start, image_size);

    // process image
    info!("Processing image...");
    let mut processed_image = ProcessedImage::new(image, 
        args.colors, args.lab);
    // if args.size is set to 0, the image will not be resized
    let size = if args.size > 0 {
        Some(Size::new(args.size, args.size))
    } else {
        None
    };
    processed_image.process(size, start_location)?;

    // save flow image
    // if output is not set, save to the same directory as the input image, with fiename {input}_flow.jpg
    let output = if args.output.is_empty() {
        let mut output = args.input.clone();
        output.push_str("_flow.jpg");
        output
    } else {
        args.output
    };
    info!("Saving flow image to: {}", output);
    let flow_image = processed_image.flow_image();
    let mut u8_image = Mat::default();
    flow_image.convert_to(&mut u8_image, opencv::core::CV_8U, 255.0, 0.0)?;
    opencv::imgcodecs::imwrite(&output, &u8_image, &opencv::core::Vector::new())?;

    // highgui::named_window(&"test", highgui::WINDOW_AUTOSIZE)?;
    // highgui::imshow(&"test", &u8_image)?;
    // highgui::wait_key(0)?;
    // highgui::destroy_all_windows()?;

    Ok(())
}