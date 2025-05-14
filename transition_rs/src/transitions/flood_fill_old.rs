use opencv::prelude::*;
use opencv::core::{Mat, Vec3b};
use std::collections::BinaryHeap;
use std::cmp::{max, Ordering};
use ordered_float::NotNan;
use ndarray::Array2;

use crate::image_processing::conversions::ndarray_f32_to_mat;

#[derive(Copy, Clone, Eq, PartialEq)]
struct HeapItem {
    dist: NotNan<f32>,
    y: i32,
    x: i32,
}

impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse order for min-heap
        other.dist.cmp(&self.dist)
    }
}

impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Create a gradient map using flood-fill starting from given points.
/// The gradient map represents the accumulated color distance from the starting points.
pub fn flood_fill_color_image(image: &Mat, start_points: &Vec<(i32, i32)>) -> opencv::Result<Mat> {
    let height = image.rows();
    let width = image.cols();

    // Initialize gradient map as an ndarray
    let mut gradient_map: ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 2]>> = Array2::<f32>::from_elem((height as usize, width as usize), f32::INFINITY);

    // Initialize heap with start points
    let mut heap = BinaryHeap::new();
    for &(y, x) in start_points.iter() {
        if y >= 0 && y < height && x >= 0 && x < width {
            heap.push(HeapItem {
                dist: NotNan::new(0.0).unwrap(),
                y,
                x,
            });
            gradient_map[(y as usize, x as usize)] = 0.0;
        }
    }

    // Direction vectors for neighboring pixels
    let directions = [(0, -1), (0, 1), (-1, 0), (1, 0)];
    
    while let Some(HeapItem { dist, y, x }) = heap.pop() {
        let dist_f32 = dist.into_inner();

        // Skip if indices are out of bounds
        if y < 0 || y >= height || x < 0 || x >= width {
            continue;
        }

        let current_gradient = gradient_map[(y as usize, x as usize)];

        if current_gradient < dist_f32 {
            continue;
        }

        // For each neighbor
        for &(dy, dx) in &directions {
            let ny = y + dy;
            let nx = x + dx;

            if ny >= 0 && ny < height && nx >= 0 && nx < width {
                let neighbor_gradient = gradient_map[(ny as usize, nx as usize)];
                let pixel_current = image.at_2d::<Vec3b>(y, x)?;
                let pixel_neighbor = image.at_2d::<Vec3b>(ny, nx)?;

                // Calculate color distance between current pixel and neighbor
                let diff = [
                    pixel_neighbor[0] as f32 - pixel_current[0] as f32,
                    pixel_neighbor[1] as f32 - pixel_current[1] as f32,
                    pixel_neighbor[2] as f32 - pixel_current[2] as f32,
                ];
                let color_distance = diff[0].powi(2) + diff[1].powi(2) + diff[2].powi(2);

                let next_dist = dist_f32 + color_distance;

                if next_dist < neighbor_gradient {
                    gradient_map[(ny as usize, nx as usize)] = next_dist;
                    heap.push(HeapItem {
                        dist: NotNan::new(next_dist).unwrap(),
                        y: ny,
                        x: nx,
                    });
                }
            }
        }
    }

    // Post-processing
    let max_dist = gradient_map.iter().cloned().filter(|v| v.is_finite()).fold(f32::MIN, f32::max);

    // Replace infinities with max_dist
    gradient_map.mapv_inplace(|v| if v.is_finite() { v } else { max_dist });

    // Normalize the gradient map to range [0, 1]
    let mut normalized_map = gradient_map.clone();
    let min_val = *normalized_map.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let max_val = *normalized_map.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();

    if max_val - min_val > f32::EPSILON {
        normalized_map.mapv_inplace(|v| (v - min_val) / (max_val - min_val));
    } else {
        normalized_map.fill(0.0);
    }

    // Convert ndarray back to OpenCV Mat
    //let normalized_mat = Mat::from_slice_2d(&normalized_map.outer_iter().map(|row| row.as_slice().unwrap()).collect::<Vec<_>>())?;
    let normalized_mat = ndarray_f32_to_mat(&normalized_map)?;

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
