// src/image_processing/gradient_maps.rs
use opencv::prelude::*;
use opencv::core::{self, Mat, Vec3b, CV_32F, CV_32FC3, NORM_MINMAX};
use opencv::core::Vector;
use opencv::imgproc::accumulate;

pub fn create_gradient_map(image: &Mat, reference_point: (i32, i32)) -> opencv::Result<Mat> {
    // Ensure the image is of type CV_8UC3
    if image.typ() != core::CV_8UC3 {
        return Err(opencv::Error::new(
            opencv::core::StsUnsupportedFormat,
            "Unsupported image format. Expected CV_8UC3.".to_string(),
        ));
    }

    // Get image dimensions
    let (height, width) = (image.rows(), image.cols());

    // Get the reference pixel value at (row, col) = (y, x)
    let ref_value = *image.at_2d::<Vec3b>(reference_point.1, reference_point.0)?;

    // Convert image to float32
    let mut image_f32 = Mat::default();
    image.convert_to(&mut image_f32, CV_32FC3, 1.0, 0.0)?;

    // Create a matrix with the reference pixel value replicated
    let ref_value_scalar = core::Scalar::new(
        ref_value[0] as f64,
        ref_value[1] as f64,
        ref_value[2] as f64,
        0.0,
    );

    let ref_mat = Mat::new_rows_cols_with_default(
        height,
        width,
        CV_32FC3,
        ref_value_scalar,
    )?;

    // Compute the absolute difference
    let mut diff_map = Mat::default();
    core::absdiff(&image_f32, &ref_mat, &mut diff_map)?;

    // Split the difference map into separate channels
    let mut channels = Vector::<Mat>::new();
    core::split(&diff_map, &mut channels)?;

    // Sum the absolute differences over the color channels
    let mut gradient_map = Mat::zeros(height, width, CV_32F)?.to_mat()?;

    // Use `core::accumulate` to sum the channels into `gradient_map`
    for channel in channels.iter() {
        accumulate(&channel, &mut gradient_map, &core::no_array())?;
    }

    // Normalize the gradient map to the range [0, 1]
    let mut normalized_map = Mat::default();
    core::normalize(
        &gradient_map,
        &mut normalized_map,
        0.0,
        1.0,
        NORM_MINMAX,
        -1,
        &core::no_array(),
    )?;

    Ok(normalized_map)
}


#[cfg(test)]
mod tests {
    use super::*;
    use opencv::core::{Scalar, CV_8UC3};

    #[test]
    fn test_create_gradient_map() -> opencv::Result<()> {
        // Create a simple test image (10x10) with a constant color
        let mut image = Mat::new_rows_cols_with_default(
            10,
            10,
            CV_8UC3,
            Scalar::new(100.0, 150.0, 200.0, 0.0),
        )?;

        // Set a different pixel value at the reference point (5, 5)
        *image.at_2d_mut::<Vec3b>(5, 5)? = Vec3b::from([50, 100, 150]);

        // Reference point is (5, 5)
        let reference_point = (5, 5);

        // Call the function
        let gradient_map = create_gradient_map(&image, reference_point)?;

        // The value at the reference point should be zero
        let value_at_ref = *gradient_map.at_2d::<f32>(5, 5)?;
        assert_eq!(value_at_ref, 0.0);

        // The value at other points should be greater than zero
        let value_at_other = *gradient_map.at_2d::<f32>(0, 0)?;
        assert!(value_at_other > 0.0);

        Ok(())
    }
}
