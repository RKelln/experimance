// src/image_processing.rs
pub mod posterization;
pub mod gradient_maps;
pub mod processed_image;
pub mod conversions;

pub use posterization::posterize_image_fast;
pub use gradient_maps::create_gradient_map;
pub use processed_image::ProcessedImage;
pub use conversions::{
    mat_to_ndarray_u8, mat_to_ndarray_u8_c3, 
    ndarray_u8_to_mat, ndarray_u8_c3_to_mat, 
    ndarray_f32_to_mat, ndarray_f32_c3_to_mat};