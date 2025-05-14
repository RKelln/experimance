use opencv::prelude::*;
use opencv::core::{Mat, Vec3b};
use std::collections::BinaryHeap;
use std::cmp::Ordering;
use ndarray::{Array2, Array3};

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

#[inline(always)]
fn color_distance(current: &[u8], neighbor: &[u8]) -> f32 {
    let dr = current[0] as i32 - neighbor[0] as i32;
    let dg = current[1] as i32 - neighbor[1] as i32;
    let db = current[2] as i32 - neighbor[2] as i32;
    (dr * dr + dg * dg + db * db) as f32
}

/// Create a gradient map using flood-fill starting from given points.
/// The gradient map represents the accumulated color distance from the starting points.
pub fn flood_fill_color_image(image: &Mat, start_points: &Vec<(i32, i32)>) -> opencv::Result<Mat> {
    let height = image.rows();
    let width = image.cols();
    let total_pixels = (height * width) as usize;

    // Use a single allocation for both gradient map and final result
    let mut gradient_map = vec![f32::MAX; total_pixels];
    let mut visited = vec![false; total_pixels];
    let mut heap = BinaryHeap::with_capacity(width.min(height) as usize * 4);
    
    // Convert image to ndarray for faster access
    let image_data: Array3<u8> = mat_to_ndarray_u8_c3(&image)?;
    let image_slice = image_data.as_slice().unwrap();
    
    // Initialize start points
    for &(x, y) in start_points {
        if y >= 0 && y < height && x >= 0 && x < width {
            let idx = (y as usize * width as usize) + x as usize;
            gradient_map[idx] = 0.0;
            heap.push(HeapItem { dist: 0.0, y, x });
        }
    }

    // Cache frequently used values
    let w = width as usize;
    let h = height as usize;
    let stride = w * 3;

    while let Some(HeapItem { dist, y, x }) = heap.pop() {
        let current_idx = (y as usize * w) + x as usize;
        if visited[current_idx] {
            continue;
        }
        visited[current_idx] = true;

        let current_pixel_idx = (y as usize * stride) + (x as usize * 3);
        let current_color = &image_slice[current_pixel_idx..current_pixel_idx + 3];

        // Check each neighbor
        let y = y as usize;
        let x = x as usize;
        
        // Unrolled neighbor checking for better performance
        if y > 0 {
            let ny = y - 1;
            let idx = ny * w + x;
            if !visited[idx] {
                let pixel_idx = ny * stride + x * 3;
                let next_dist = dist + color_distance(current_color, 
                    &image_slice[pixel_idx..pixel_idx + 3]);
                if next_dist < gradient_map[idx] {
                    gradient_map[idx] = next_dist;
                    heap.push(HeapItem { dist: next_dist, y: ny as i32, x: x as i32 });
                }
            }
        }
        
        if y < h - 1 {
            let ny = y + 1;
            let idx = ny * w + x;
            if !visited[idx] {
                let pixel_idx = ny * stride + x * 3;
                let next_dist = dist + color_distance(current_color, 
                    &image_slice[pixel_idx..pixel_idx + 3]);
                if next_dist < gradient_map[idx] {
                    gradient_map[idx] = next_dist;
                    heap.push(HeapItem { dist: next_dist, y: ny as i32, x: x as i32 });
                }
            }
        }
        
        if x > 0 {
            let nx = x - 1;
            let idx = y * w + nx;
            if !visited[idx] {
                let pixel_idx = y * stride + nx * 3;
                let next_dist = dist + color_distance(current_color, 
                    &image_slice[pixel_idx..pixel_idx + 3]);
                if next_dist < gradient_map[idx] {
                    gradient_map[idx] = next_dist;
                    heap.push(HeapItem { dist: next_dist, y: y as i32, x: nx as i32 });
                }
            }
        }
        
        if x < w - 1 {
            let nx = x + 1;
            let idx = y * w + nx;
            if !visited[idx] {
                let pixel_idx = y * stride + nx * 3;
                let next_dist = dist + color_distance(current_color, 
                    &image_slice[pixel_idx..pixel_idx + 3]);
                if next_dist < gradient_map[idx] {
                    gradient_map[idx] = next_dist;
                    heap.push(HeapItem { dist: next_dist, y: y as i32, x: nx as i32 });
                }
            }
        }
    }

    // Find max value for normalization (excluding MAX values)
    let max_dist = gradient_map.iter()
        .filter(|&&x| x < f32::MAX)
        .fold(0.0f32, |max, &x| max.max(x));

    if max_dist > 0.0 {
        gradient_map.iter_mut().for_each(|x| {
            if *x == f32::MAX {
                *x = 1.0;
            } else {
                *x /= max_dist;
            }
        });
    }

    // Convert back to Mat
    let gradient_map_2d = Array2::from_shape_vec((h, w), gradient_map)
        .expect("Failed to reshape gradient map");
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
