// src/image_processing/posterization.rs
use opencv::core::{Mat, Vec3b, CV_32FC3, CV_8UC3};
use opencv::prelude::*;


/// Posterizes an image by reducing the number of colors per channel.
/// This method is significantly faster than using KMeans clustering.
pub fn posterize_image_fast_kernel(image: &Mat, num_bits: usize) -> opencv::Result<Mat> {
    let shift = 8 - num_bits;
    let scale = 1 << shift;

    let mut posterized = Mat::default();
    
    // Convert image to float for processing
    image.convert_to(&mut posterized, opencv::core::CV_32FC3, (1.0 / scale as f32).into(), 0.0)?;
    
    // Reduce the number of colors by flooring the values
    let kernel = Mat::from_slice_2d(&[
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])?;
    let mut transformed = Mat::default();
    opencv::core::transform(&posterized, &mut transformed, &kernel)?;
    
    // Scale back to original range
    opencv::core::multiply(&transformed, &Mat::from_slice(&[scale as f32])?, &mut posterized, 1.0, -1)?;

    Ok(posterized)
}

/// Posterizes an image by reducing the number of colors per channel.
pub fn posterize_image_fast(image: &Mat, num_bits: usize) -> opencv::Result<Mat> {
    // Retrieve the size of the image
    let size = image.size()?; // size is of type Size_<i32>

    // Access width and height from the Size_<i32> struct
    let cols = size.width;
    let rows = size.height;
    
    let mut posterized = Mat::default();
    image.clone_into(&mut posterized);

    for y in 0..rows {
        for x in 0..cols {
            let pixel = image.at_2d::<Vec3b>(y as i32, x as i32)?;
            let r = (pixel[0] >> (8 - num_bits)) << (8 - num_bits);
            let g = (pixel[1] >> (8 - num_bits)) << (8 - num_bits);
            let b = (pixel[2] >> (8 - num_bits)) << (8 - num_bits);
            *posterized.at_2d_mut::<Vec3b>(y as i32, x as i32)? = Vec3b::from([b as u8, g as u8, r as u8]);
        }
    }

    Ok(posterized)
}

#[cfg(test)]
mod tests {
    use super::*;
    use opencv::core::{Scalar, Vec3b, CV_8UC3};
    
    #[test]
    fn test_posterize_image_fast_zeros() -> opencv::Result<()> {
        let rows = 4;
        // Create a 2x2 image with all zeros
        let image = Mat::new_rows_cols_with_default(rows, rows, CV_8UC3, Scalar::new(0.0, 0.0, 0.0, 0.0))?;
        // Apply posterization with 2 colors
        let posterized_image = posterize_image_fast(&image, 2)?;
        // Verify dimensions
        assert_eq!(posterized_image.rows(), rows);
        assert_eq!(posterized_image.cols(), rows);
        // Additional checks can be added here
        Ok(())
    }

    #[test]
    fn test_posterize_image_fast_colors() -> opencv::Result<()> {
        // Create a 2x2 image with varying colors
        let mut image = Mat::new_rows_cols_with_default(2, 2, CV_8UC3, Scalar::new(0.0, 0.0, 0.0, 0.0))?;

        // Modify the image data
        *image.at_2d_mut::<Vec3b>(0, 0)? = Vec3b::from([10, 20, 30]);
        *image.at_2d_mut::<Vec3b>(0, 1)? = Vec3b::from([40, 50, 60]);
        *image.at_2d_mut::<Vec3b>(1, 0)? = Vec3b::from([70, 80, 90]);
        *image.at_2d_mut::<Vec3b>(1, 1)? = Vec3b::from([100, 110, 120]);

        // Apply posterization with 2 colors
        let posterized_image = posterize_image_fast(&image, 2)?;

        // Verify dimensions
        assert_eq!(posterized_image.rows(), 2);
        assert_eq!(posterized_image.cols(), 2);

        // Verify that the posterized image has the expected number of unique colors
        let mut unique_colors = std::collections::HashSet::new();
        for y in 0..posterized_image.rows() {
            for x in 0..posterized_image.cols() {
                let pixel = *posterized_image.at_2d::<Vec3b>(y, x)?;
                unique_colors.insert([pixel[0], pixel[1], pixel[2]]);
            }
        }
        assert!(unique_colors.len() <= 2);

        Ok(())
    }
}
