use opencv::prelude::*;
use opencv::core::{Mat, Vec3b};
use std::collections::BinaryHeap;
use std::cmp::Ordering;
use ndarray::{Array1, Array2, Array3};

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

/// Create a gradient map using flood-fill starting from given points.
/// The gradient map represents the accumulated color distance from the starting points.
pub fn flood_fill_color_image(image: &Mat, start_points: &Vec<(i32, i32)>) -> opencv::Result<Mat> {
    let height = image.rows();
    let width = image.cols();
    let total_pixels = (height * width) as usize;

    // Initialize gradient map with 0.0 instead of MAX
    let mut gradient_map = vec![0.0; total_pixels];
    let mut visited = vec![false; total_pixels];
    let mut heap = BinaryHeap::new();
    
    let image_data: Array3<u8> = mat_to_ndarray_u8_c3(&image)?;
    let mut max_dist = 0.0f32;
    
    // Initialize all non-start points with a very large value
    gradient_map.fill(f32::MAX);
    
    // Initialize start points
    for &(x, y) in start_points {
        if y >= 0 && y < height && x >= 0 && x < width {
            let idx = (y as usize * width as usize) + x as usize;
            gradient_map[idx] = 0.0;
            heap.push(HeapItem { dist: 0.0, y, x });
        }
    }

    let directions = [(0, -1), (0, 1), (-1, 0), (1, 0)];
    let mut pixels_visited = 0;

    while let Some(HeapItem { dist, y, x }) = heap.pop() {
        let current_idx = (y as usize * width as usize) + x as usize;
        
        if visited[current_idx] {
            continue;
        }
        
        visited[current_idx] = true;
        pixels_visited += 1;
        max_dist = max_dist.max(dist);

        // For each neighbor
        for &(dy, dx) in &directions {
            let ny = y + dy;
            let nx = x + dx;

            if ny < 0 || ny >= height || nx < 0 || nx >= width {
                continue;
            }

            let neighbor_idx = (ny as usize * width as usize) + nx as usize;
            if visited[neighbor_idx] {
                continue;
            }

            // Calculate squared color distance
            let dr = image_data[[y as usize, x as usize, 0]] as i32 
                  - image_data[[ny as usize, nx as usize, 0]] as i32;
            let dg = image_data[[y as usize, x as usize, 1]] as i32 
                  - image_data[[ny as usize, nx as usize, 1]] as i32;
            let db = image_data[[y as usize, x as usize, 2]] as i32 
                  - image_data[[ny as usize, nx as usize, 2]] as i32;
            
            let color_distance = (dr * dr + dg * dg + db * db) as f32;
            let next_dist = dist + color_distance;

            if next_dist < gradient_map[neighbor_idx] {
                gradient_map[neighbor_idx] = next_dist;
                heap.push(HeapItem {
                    dist: next_dist,
                    y: ny,
                    x: nx,
                });
            }
        }
    }

    // If we haven't visited all pixels, do a second pass with manhattan distance
    if pixels_visited < total_pixels {
        for y in 0..height {
            for x in 0..width {
                let idx = (y as usize * width as usize) + x as usize;
                if !visited[idx] {
                    // Find nearest visited pixel using manhattan distance
                    let mut min_dist = f32::MAX;
                    for &(start_x, start_y) in start_points {
                        let manhattan = ((x - start_x).abs() + (y - start_y).abs()) as f32;
                        min_dist = min_dist.min(manhattan);
                    }
                    gradient_map[idx] = max_dist * (min_dist / (width + height) as f32);
                }
            }
        }
    }

    // Normalize using tracked max_dist
    if max_dist > 0.0 {
        gradient_map.iter_mut().for_each(|x| *x /= max_dist);
    }

    // Convert back to Mat
    let gradient_map_2d = Array2::from_shape_vec((height as usize, width as usize), gradient_map)
        .expect("Failed to reshape gradient map");
    let normalized_mat = ndarray_f32_to_mat(&gradient_map_2d)?;

    Ok(normalized_mat)
}

// ...existing tests...
