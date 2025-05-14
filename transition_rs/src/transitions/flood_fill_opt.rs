use opencv::prelude::*;
use opencv::core::{Mat, Vec3b};
use std::collections::BinaryHeap;
use std::cmp::Ordering;
use ndarray::{Array1, Array2, Array3};
use ndarray_stats::QuantileExt; // Import the trait for min and max

use crate::image_processing::conversions::ndarray_f32_to_mat;
use crate::image_processing::mat_to_ndarray_u8_c3;

#[derive(Copy, Clone, PartialEq)]
struct HeapItem {
    dist: f32,
    y: i32,
    x: i32,
}

impl Eq for HeapItem {}

impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse order for min-heap
        other.dist.partial_cmp(&self.dist).unwrap()
    }
}

impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub fn minmax_normalize(arr : &mut Array1<f32>) {
    // Compute the minimum and maximum values in the gradient_map
    let (min, max) = {
        let min = *arr.min().unwrap();
        let max = *arr.max().unwrap();
        (min, max)
    };
    let diff = max - min;
    // Prevent division by zero in case all elements are the same
    if (diff).abs() < std::f32::EPSILON {
        return;
    }

    // Normalize the gradient map to range [0, 1] using mapv
    arr.mapv_inplace(|v| (v - min) / (diff));
}

pub fn minmax_normalize_2(arr : &mut Array1<f32>) {
    // Normalize the gradient map to range [0, 1]
    let min_dist = *arr.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let max_dist = *arr.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let diff_dist = max_dist - min_dist;
    if diff_dist > f32::EPSILON {
        arr.mapv_inplace(|v| (v - min_dist) / (diff_dist));
        //arr.iter_mut().for_each(|v| *v = (*v - min_dist) / diff_dist);
    } else {
        arr.fill(0.0);
    }
}

/// Create a gradient map using flood-fill starting from given points.
/// The gradient map represents the accumulated color distance from the starting points.
pub fn flood_fill_color_image(image: &Mat, start_points: &Vec<(i32, i32)>) -> opencv::Result<Mat> {
    let height = image.rows();
    let width = image.cols();
    let total_pixels = (height * width) as usize;

    // Initialize gradient map as a flat ndarray with squared distances set to infinity
    let mut gradient_map = Array1::<f32>::from_elem(total_pixels, f32::INFINITY);

    // Initialize heap with start points
    let mut heap = BinaryHeap::new();
    for &(y, x) in start_points.iter() {
        if y >= 0 && y < height && x >= 0 && x < width {
            heap.push(HeapItem {
                dist: 0.0,
                y,
                x,
            });
            //gradient_map[(y as usize, x as usize)] = 0.0;
            let idx = (y as usize * width as usize) + (x as usize);
            gradient_map[idx] = 0.0;
        }
    }

    let image_data: Array3<u8> = mat_to_ndarray_u8_c3(&image)?;
    
    // Direction vectors for neighboring pixels
    let directions = [(0, -1), (0, 1), (-1, 0), (1, 0)];

    while let Some(HeapItem { dist, y, x }) = heap.pop() {

        // Skip if indices are out of bounds
        if y < 0 || y >= height || x < 0 || x >= width {
            continue;
        }

        let new_idx = (y as usize * width as usize) + (x as usize);
        let current_gradient = gradient_map[new_idx];
        //let current_gradient = gradient_map[(y as usize, x as usize)];

        if current_gradient < dist {
            continue;
        }

        // For each neighbor
        for &(dy, dx) in &directions {
            let ny = y + dy;
            let nx = x + dx;

            // Boundary check
            if ny < 0 || ny >= height || nx < 0 || nx >= width {
                continue;
            }

            let idx = (ny as usize * width as usize) + (nx as usize);
            let neighbor_gradient = gradient_map[idx];

            // Calculate squared color distance (Euclidean)
            //println!("y: {}, x: {}, ny: {}, nx: {}", y, x, ny, nx);
            let r_current = image_data[[y as usize, x as usize, 0]] as f32;
            let g_current = image_data[[y as usize, x as usize, 1]] as f32;
            let b_current = image_data[[y as usize, x as usize, 2]] as f32;

            let r_neighbor = image_data[[ny as usize, nx as usize, 0]] as f32;
            let g_neighbor = image_data[[ny as usize, nx as usize, 1]] as f32;
            let b_neighbor = image_data[[ny as usize, nx as usize, 2]] as f32;

            let color_distance = (r_current - r_neighbor).powi(2)
                + (g_current - g_neighbor).powi(2)
                + (b_current - b_neighbor).powi(2);

            let next_dist = dist + color_distance;

            if next_dist < neighbor_gradient {
                gradient_map[idx] = next_dist;
                heap.push(HeapItem {
                    dist: next_dist,
                    y: ny,
                    x: nx,
                });
            }
        }
    }

    // Post-processing
    //let max_dist = gradient_map.iter().cloned().filter(|v| v.is_finite()).fold(f32::MIN, f32::max);

    // Replace infinities with max_dist
    //gradient_map.mapv_inplace(|v| if v.is_finite() { v } else { max_dist });

    //minmax_normalize_2(&mut gradient_map);

    // Normalize the gradient map to range [0, 1]
    minmax_normalize_2(&mut gradient_map);

    // reshape 1D gradient map to 2D
    let shape = (height as usize, width as usize);
    let gradient_map_2d = Array2::from_shape_vec(shape, gradient_map.to_vec()).unwrap();
    let normalized_mat = ndarray_f32_to_mat(&gradient_map_2d)?;

    Ok(normalized_mat)
}


#[cfg(test)]
mod tests {
    use super::*;
    use opencv::core::{Scalar, Vec3b, CV_8UC3};


    #[test]
    fn test_flood_fill_color_image() -> opencv::Result<()> {
        // Create a 5x5 image with varying colors
        let mut image = Mat::new_rows_cols_with_default(5, 5, CV_8UC3, Scalar::all(0.0))?;

        // Set pixel values
        for y in 0..5 {
            for x in 0..5 {
                *image.at_2d_mut::<Vec3b>(y, x)? = Vec3b::from([
                    (x * 10 + y * 10) as u8,
                    (x * 5 + y * 5) as u8,
                    (x * 2 + y * 2) as u8,
                ]);
            }
        }

        // Starting point at center
        let start_points = vec![(2, 2)];

        // Call the function
        let gradient_map = flood_fill_color_image(&image, &start_points)?;

        // Check that gradient_map values are between 0 and 1
        let data = gradient_map.data_typed::<f32>()?;
        for &v in data {
            assert!(v >= 0.0 && v <= 1.0);
        }

        Ok(())
    }
}
