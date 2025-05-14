use opencv::boxed_ref::BoxedRef;
use opencv::prelude::*;
use opencv::core::{Mat, CV_32F, CV_32FC3, CV_8UC1, CV_8UC3};
use ndarray::{Array2, Array3};
use ndarray::array;

/// Converts an ndarray Array2<u8> to an OpenCV Mat of type CV_8UC1.
pub fn ndarray_u8_to_mat(array: &Array2<u8>) -> opencv::Result<Mat> {
    let (rows, cols) = array.dim();

    // Ensure the array is in standard layout
    let array_std = array.as_standard_layout();

    // Get a slice to the data
    let data = array_std.as_slice().ok_or_else(|| {
        opencv::Error::new(opencv::core::StsError, "Failed to get data slice".to_string())
    })?;

    // Create a Mat and copy data
    let mut mat = Mat::zeros(rows as i32, cols as i32, CV_8UC1)?.to_mat()?;
    let mat_data = mat.data_bytes_mut()?;
    mat_data.copy_from_slice(data);

    Ok(mat)
}

/// Converts an ndarray Array3<u8> to an OpenCV Mat of type CV_8UC3.
pub fn ndarray_u8_c3_to_mat(array: &Array3<u8>) -> opencv::Result<Mat> {
    let (rows, cols, channels) = array.dim();

    if channels != 3 {
        return Err(opencv::Error::new(
            opencv::core::StsError,
            format!("Expected 3 channels, but got {}", channels),
        ));
    }

    // Ensure the array is in standard layout
    let array_std = array.as_standard_layout();

    // Get a slice to the data
    let data = array_std.as_slice().ok_or_else(|| {
        opencv::Error::new(opencv::core::StsError, "Failed to get data slice".to_string())
    })?;

    // Create a Mat and copy data
    let mut mat = Mat::zeros(rows as i32, cols as i32, CV_8UC3)?.to_mat()?;
    let mat_data = mat.data_bytes_mut()?;
    mat_data.copy_from_slice(data);

    Ok(mat)
}

/// Converts a 3D ndarray of u8 (with 3 channels) to a color OpenCV Mat with type CV_8UC3.
///
/// # Arguments
///
/// * `array` - Reference to a 3D ndarray containing u8 values. Expected shape: (rows, cols, channels).
///
/// # Returns
///
/// * `opencv::Result<Mat>` - OpenCV Mat with 3 channels of 8-bit unsigned integers.
// pub fn ndarray_u8_c3_to_mat(array: &Array3<u8>) -> opencv::Result<Mat> {
//     let (rows, cols, channels) = array.dim();

//     // Ensure the array has exactly 3 channels
//     if channels != 3 {
//         return Err(opencv::Error::new(
//             opencv::core::StsBadArg,
//             format!("Expected 3 channels, found {}", channels),
//         ));
//     }

//     // Ensure the array is contiguous in memory
//     if !array.is_standard_layout() {
//         return Err(opencv::Error::new(
//             opencv::core::StsBadArg,
//             "Input array is not contiguous in memory",
//         ));
//     }

//     // Get a slice of the data
//     let data = array.as_slice().ok_or_else(|| {
//         opencv::Error::new(
//             opencv::core::StsBadArg,
//             "Failed to get slice from ndarray::Array3<u8>",
//         )
//     })?;

//     // Create a Mat from the data slice and convert BoxedRef<Mat> to Mat
//     unsafe {
//         Mat::new_rows_cols_with_data(
//             rows as i32,
//             cols as i32,
//             data,
//         ).map(|boxed_ref| *boxed_ref)
//     }
// }

pub fn ndarray_f32_to_mat(array: &Array2<f32>) -> opencv::Result<Mat> {
    let (rows, cols) = array.dim();

    // Ensure the array is in standard layout
    let array_std = array.as_standard_layout();

    // Get a slice to the data
    let data = array_std.as_slice().ok_or_else(|| {
        opencv::Error::new(opencv::core::StsError, "Failed to get data slice".to_string())
    })?;

    // Create a Mat and copy data
    let mut mat = Mat::zeros(rows as i32, cols as i32, CV_32F)?.to_mat()?;
    let mat_data = mat.data_typed_mut::<f32>()?;
    mat_data.copy_from_slice(data);

    Ok(mat)
}

pub fn ndarray_f32_c3_to_mat(array: &Array3<f32>) -> opencv::Result<Mat> {
    let (rows, cols, channels) = array.dim();

    if channels != 3 {
        return Err(opencv::Error::new(
            opencv::core::StsError,
            format!("Expected 3 channels, but got {}", channels),
        ));
    }

    // Ensure the array is in standard layout
    let array_std = array.as_standard_layout();

    // Get a slice to the data
    let data = array_std.as_slice().ok_or_else(|| {
        opencv::Error::new(opencv::core::StsError, "Failed to get data slice".to_string())
    })?;

    // Create a Mat and copy data
    let mut mat = Mat::zeros(rows as i32, cols as i32, CV_32FC3)?.to_mat()?;
    let mat_data = mat.data_typed_mut::<f32>()?;
    mat_data.copy_from_slice(data);

    Ok(mat)
}

pub fn mat_to_ndarray_u8(mat: &Mat) -> opencv::Result<Array2<u8>> {
    let rows = mat.rows();
    let cols = mat.cols();

    let mut array = Array2::<u8>::zeros((rows as usize, cols as usize));
    let array_data = array.as_slice_mut().ok_or_else(|| {
        opencv::Error::new(opencv::core::StsError, "Failed to get data slice".to_string())
    })?;

    let mat_data = mat.data_bytes()?;
    array_data.copy_from_slice(mat_data);

    Ok(array)
}

pub fn mat_to_ndarray_u8_c3(mat: &Mat) -> opencv::Result<Array3<u8>> {
    let rows = mat.rows();
    let cols = mat.cols();

    let mut array = Array3::<u8>::zeros((rows as usize, cols as usize, 3));
    let array_data = array.as_slice_mut().ok_or_else(|| {
        opencv::Error::new(opencv::core::StsError, "Failed to get data slice".to_string())
    })?;

    let mat_data = mat.data_bytes()?;
    array_data.copy_from_slice(mat_data);

    Ok(array)
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ndarray_u8_to_mat() {
        let array = array![[1, 2, 3], [4, 5, 6], [7, 8, 9]];
        let mat = ndarray_u8_to_mat(&array).unwrap();
        assert_eq!(mat.at_2d::<u8>(0, 0).unwrap(), &1);
        assert_eq!(mat.at_2d::<u8>(1, 1).unwrap(), &5);
        assert_eq!(mat.at_2d::<u8>(2, 2).unwrap(), &9);
    }

    #[test]
    fn test_ndarray_u8_c3_to_mat() {
        let array = array![
            [[1, 2, 3], [4, 5, 6]],
            [[7, 8, 9], [10, 11, 12]]
        ];
        let mat = ndarray_u8_c3_to_mat(&array).unwrap();

        // Verify Mat properties
        assert_eq!(mat.rows(), 2);
        assert_eq!(mat.cols(), 2);
        assert_eq!(mat.typ(), CV_8UC3);
        assert_eq!(mat.channels(), 3);

        // Verify individual elements
        assert_eq!(*mat.at_3d::<u8>(0, 0, 0).unwrap(), 1);
        assert_eq!(*mat.at_3d::<u8>(0, 0, 1).unwrap(), 2);
        assert_eq!(*mat.at_3d::<u8>(0, 0, 2).unwrap(), 3);

        assert_eq!(*mat.at_3d::<u8>(1, 1, 0).unwrap(), 10);
        assert_eq!(*mat.at_3d::<u8>(1, 1, 1).unwrap(), 11);
        assert_eq!(*mat.at_3d::<u8>(1, 1, 2).unwrap(), 12);
    }

    #[test]
    fn test_ndarray_f32_to_mat() {
        let array = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let mat = ndarray_f32_to_mat(&array).unwrap();
        assert_eq!(mat.at_2d::<f32>(0, 0).unwrap(), &1.0);
        assert_eq!(mat.at_2d::<f32>(1, 1).unwrap(), &5.0);
        assert_eq!(mat.at_2d::<f32>(2, 2).unwrap(), &9.0);
    }

    #[test]
    fn test_ndarray_f32_c3_to_mat() {
        let array = array![
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]
        ];
        let mat = ndarray_f32_c3_to_mat(&array).unwrap();
        assert_eq!(mat.at_3d::<f32>(0, 0, 0).unwrap(), &1.0);
        assert_eq!(mat.at_3d::<f32>(1, 1, 2).unwrap(), &12.0);
    }

    #[test]
    fn test_mat_to_ndarray_u8() {
        let array = array![[1, 2, 3], [4, 5, 6], [7, 8, 9]];
        let mat = ndarray_u8_to_mat(&array).unwrap();
        let ndarray = mat_to_ndarray_u8(&mat).unwrap();
        assert_eq!(ndarray, array);
    }

    #[test]
    fn test_mat_to_ndarray_u8_c3() {
        let array = array![
            [[1, 2, 3], [4, 5, 6]],
            [[7, 8, 9], [10, 11, 12]]
        ];
        let mat = ndarray_u8_c3_to_mat(&array).unwrap();
        let ndarray = mat_to_ndarray_u8_c3(&mat).unwrap();
        assert_eq!(ndarray, array);
    }
}
