use opencv::core::{Mat, CV_32FC1, CV_8UC3};
use ordered_float::OrderedFloat;
use std::hash::{Hash, Hasher};
use opencv::prelude::*;
use priority_queue::PriorityQueue;
use rayon::prelude::*;
use ndarray::Array3;

use crate::image_processing::mat_to_ndarray_u8_c3;

// Define the HeapItem struct with squared distance
#[derive(Debug, Clone, Copy)]
struct HeapItem {
    y: i32,
    x: i32,
    dist_sq: f32,
}

impl Hash for HeapItem {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.y.hash(state);
        self.x.hash(state);
        OrderedFloat(self.dist_sq).hash(state);
    }
}

impl PartialEq for HeapItem {
    fn eq(&self, other: &Self) -> bool {
        self.dist_sq.eq(&other.dist_sq)
    }
}

impl Eq for HeapItem {}

impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.dist_sq.partial_cmp(&other.dist_sq).unwrap())
    }
}

impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.dist_sq.partial_cmp(&self.dist_sq).unwrap()
    }
}


/// Convert a flat gradient_map_sq vector to a single-channel Mat
fn gradient_map_to_mat_flat(gradient_map_sq: &[f32], width: usize, height: i32) -> opencv::Result<Mat> {
    // Ensure the gradient_map_sq has exactly width * height elements
    assert_eq!(gradient_map_sq.len(), (width * height as usize) as usize);

    // Create a Mat from the flat slice
    let gradient_flat_mat = Mat::from_slice(&gradient_map_sq)?;

    // Reshape the Mat to have one channel and the specified height
    let gradient_mat = gradient_flat_mat.reshape(1, height)?;

    // Convert to CV_32FC1
    let mut final_gradient = Mat::default();
    gradient_mat.convert_to(&mut final_gradient, CV_32FC1, 1.0, 0.0)?;

    Ok(final_gradient)
}


/// Create a gradient map using flood-fill starting from given points.
/// The gradient map represents the accumulated color distance from the starting points.
///
/// # Arguments
///
/// * `image` - Reference to the source image as an OpenCV Mat.
/// * `start_points` - Vector of starting points (y, x) for the flood-fill.
///
/// # Returns
///
/// * `opencv::Result<Mat>` - Gradient map as a single-channel 32-bit float Mat.
pub fn flood_fill_color_image_optimized(
    image: &Mat,
    start_points: &Vec<(i32, i32)>,
) -> opencv::Result<Mat> {
    let height = image.rows();
    let width = image.cols();
    let total_pixels = (height * width) as usize;

    // Initialize gradient map as a flat vector with squared distances set to infinity
    let mut gradient_map_sq = vec![f32::INFINITY; total_pixels];

    // Initialize priority queue (min-heap) with starting points
    let mut heap = PriorityQueue::new();
    for &(y, x) in start_points {
        if y >= 0 && y < height && x >= 0 && x < width {
            let idx = (y as usize * width as usize) + (x as usize);
            gradient_map_sq[idx] = 0.0;
            heap.push(
                HeapItem {
                    y,
                    x,
                    dist_sq: 0.0,
                },
                OrderedFloat(0.0), // Priority is dist_sq; lower value means higher priority
            );
        }
    }

    // Define the 8-connected neighbor directions
    let directions = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    ];

    // Convert Mat to ndarray for faster pixel access
    let image_data: Array3<u8> = mat_to_ndarray_u8_c3(image)?;

    // Optional: Define a maximum distance threshold to limit processing
    let max_distance_sq: f32 = 10000000.0; // Example: 100^2

    // Flood-fill algorithm
    while let Some((current, _)) = heap.pop() {
        let current_idx = (current.y as usize * width as usize) + (current.x as usize);
        let current_dist_sq = gradient_map_sq[current_idx];

        // Early termination if the current distance exceeds the maximum threshold
        if current_dist_sq > max_distance_sq {
            continue;
        }

        for &(dy, dx) in &directions {
            let new_y = current.y + dy;
            let new_x = current.x + dx;

            // Boundary check
            if new_y < 0 || new_y >= height || new_x < 0 || new_x >= width {
                continue;
            }

            let new_idx = (new_y as usize * width as usize) + (new_x as usize);
            let neighbor_dist_sq = gradient_map_sq[new_idx];

            // Calculate squared color distance (Euclidean)
            let r_current = image_data[[current.y as usize, current.x as usize, 0]] as f32;
            let g_current = image_data[[current.y as usize, current.x as usize, 1]] as f32;
            let b_current = image_data[[current.y as usize, current.x as usize, 2]] as f32;

            let r_neighbor = image_data[[new_y as usize, new_x as usize, 0]] as f32;
            let g_neighbor = image_data[[new_y as usize, new_x as usize, 1]] as f32;
            let b_neighbor = image_data[[new_y as usize, new_x as usize, 2]] as f32;

            let color_distance_sq = (r_current - r_neighbor).powi(2)
                + (g_current - g_neighbor).powi(2)
                + (b_current - b_neighbor).powi(2);

            let new_dist_sq = current_dist_sq + color_distance_sq;

            // Update if a shorter distance is found and within the threshold
            if new_dist_sq < neighbor_dist_sq && new_dist_sq <= max_distance_sq {
                gradient_map_sq[new_idx] = new_dist_sq;
                heap.push(
                    HeapItem {
                        y: new_y,
                        x: new_x,
                        dist_sq: new_dist_sq,
                    },
                    OrderedFloat(new_dist_sq), // Priority is dist_sq; lower value has higher priority
                );
            }
        }
    }


    // Convert the gradient_map_sq back to a single-channel Mat using the helper function
    let final_gradient = gradient_map_to_mat_flat(&gradient_map_sq, width as usize, height)?;

    Ok(final_gradient)
}

#[cfg(test)]
mod tests {
    use super::*;
    use opencv::core::{Scalar, Vec3b, CV_8UC3, CV_32FC1};
    use opencv::prelude::*;

    #[test]
    fn test_flood_fill_color_image_optimized() -> opencv::Result<()> {
        // Create a simple 5x5 image with distinct colors
        let mut image = Mat::new_rows_cols_with_default(5, 5, CV_8UC3, Scalar::all(0.0))?;

        // Assign colors to the image
        for y in 0..5 {
            for x in 0..5 {
                let color = Vec3b::from([x as u8 * 50, y as u8 * 50, 100]);
                // Directly assign the color to avoid type mismatch
                *image.at_2d_mut::<Vec3b>(y, x)? = color;
            }
        }

        // Define starting points
        let start_points = vec![(2, 2)];

        // Perform flood-fill
        let gradient_mat = flood_fill_color_image_optimized(&image, &start_points)?;

        // Check the size and type of the gradient map
        assert_eq!(gradient_mat.rows(), 5);
        assert_eq!(gradient_mat.cols(), 5);
        assert_eq!(gradient_mat.typ(), CV_32FC1);

        // Optionally, print the gradient map for visual inspection
        for y in 0..5 {
            for x in 0..5 {
                let val = *gradient_mat.at_2d::<f32>(y, x)?;
                print!("{:.1} ", val);
            }
            println!();
        }

        Ok(())
    }
}