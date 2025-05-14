// src/image_processing/processed_image.rs
use opencv::core::{Mat, MatTraitConst, Size, Vec3b};
use opencv::imgproc::{cvt_color, resize, COLOR_BGR2Lab, InterpolationFlags};
use opencv::Result;

use crate::transitions::flood_fill_opt3::flood_fill_color_image;
//use crate::transitions::flood_fill_old::flood_fill_color_image;
use super::posterize_image_fast;
use opencv::imgcodecs::{imread, IMREAD_COLOR};
use opencv::core::{self, min_max_loc};

use crate::time_it; 
use log::info;
use log::warn;
use std::time::Instant;

pub struct ProcessedImage {
    image: Mat,
    resized_image: Option<Mat>,
    posterized_image: Option<Mat>,
    flow: Option<Mat>,
    pub poster_colors: usize,
    pub convert_to_lab: bool,
}

impl ProcessedImage {
    pub fn new(image: Mat, poster_colors: usize, convert_to_lab: bool) -> Self {
        
        // Convert to LAB color space if required
        let _image = if convert_to_lab {
            Self::convert_to_lab(&image).unwrap()
        } else {
            image // Move ownership without cloning
        };

        ProcessedImage {
            image: _image,
            resized_image: None,
            posterized_image: None,
            flow: None,
            poster_colors,
            convert_to_lab,
        }
    }

    // Getter for `image`
    pub fn image(&self) -> &Mat {
        &self.image
    }

    pub fn posterized_image(&self) -> &Mat {
        self.posterized_image.as_ref().unwrap()
    }

    pub fn flow_image(&self) -> &Mat {
        self.flow.as_ref().unwrap()
    }

    // Setter for `image`
    pub fn set_image(&mut self, image: Mat) {
        self.image = image;
        // reset the saved processed images
        self.resized_image = None;
        self.posterized_image = None;
        self.flow = None;

        if self.convert_to_lab {
            self.image = Self::convert_to_lab(&self.image).unwrap();
        }
    }

    // Associated Function: Convert image to LAB color space
    fn convert_to_lab(image: &Mat) -> Result<Mat> {
        let mut lab_image = Mat::default();
        cvt_color(image, &mut lab_image, COLOR_BGR2Lab, 0)?;
        Ok(lab_image)
    }


    pub fn process(&mut self, size: Option<Size>, location: Option<Vec<(i32, i32)>>) -> opencv::Result<()> {
        assert!(self.image.size().is_ok() && self.image.size().unwrap().width > 0 && self.image.size().unwrap().height > 0);
        time_it!("resize", {self.resize(size)?});
        //time_it!("posterize", {self.posterize()?});
        time_it!("flow", {self.flow_map(location)?});
        Ok(())
    }


    fn resize_image(image: &Mat, new_size: Option<Size>) -> Result<Mat> {
        let resized_image = if let Some(new_size) = new_size {
            let mut resized = Mat::default();
            resize(
                image,
                &mut resized,
                new_size,
                0.0,
                0.0,
                InterpolationFlags::INTER_AREA.into(),
            )?;
            resized
        } else {
            image.clone()
        };
    
        Ok(resized_image)
    }

    fn resize(&mut self, _size: Option<Size>) -> opencv::Result<()> {

        // only resize once
        if self.resized_image.is_some() {
            return Ok(());
        }

        // if size is None or size[0] == 0 or size[1] == 0 or size == self.image.shape[:2]: # no resizing
        if _size.is_none() || self.image.size().unwrap() == _size.unwrap() {
            self.resized_image = Some(self.image.clone());
        }
        else {
            let size = _size.unwrap();
            self.resized_image = Some(Self::resize_image(&self.image, Some(size))?);
        }

        Ok(())
    }

    fn posterize(&mut self) -> opencv::Result<()> {
    
        // only posterize once
        if self.posterized_image.is_some() {
            return Ok(());
        }

        if self.poster_colors == 0 || self.poster_colors >= 128 {
            self.posterized_image = Some(self.resized_image.clone().unwrap());
        } else {
            // posterize
            // convert num_colors to num_bits (min: 1)
            let num_bits = (self.poster_colors as f32).log2().ceil() as usize;
            self.posterized_image = Some(posterize_image_fast(&self.resized_image.clone().unwrap(), num_bits)?);
        } 

        Ok(())
    }

    fn flow_map(&mut self, _locations: Option<Vec<(i32, i32)>>) -> opencv::Result<()> {
        
        // only create flow map once
        if self.flow.is_some() {
            return Ok(());
        }

        // if posterized image isn't set then use resized image instead
        if self.posterized_image.is_none() {
            self.posterized_image = Some(self.resized_image.clone().unwrap());
        }

        let resized_image = self.resized_image.as_ref().unwrap();
        let (width, height) = (resized_image.cols(), resized_image.rows());
        
        let locations = if let Some(locations) = _locations {
            // Validate each location is within bounds
            locations.into_iter()
                .map(|(x, y)| (
                    x.clamp(0, width - 1),
                    y.clamp(0, height - 1)
                ))
                .collect()
        } else {
            vec![(width / 2, height / 2)]
        };
        let start = Instant::now();

        info!("Creating flow map with start location(s): {:?} for image size: {:?}", locations, (width, height));

        let flow = flood_fill_color_image(&self.posterized_image.clone().unwrap(), &locations)?;
        let duration = start.elapsed();

        // Log flow map statistics using minMaxLoc
        let mut min_val = 0f64;
        let mut max_val = 0f64;
        let mut min_loc = core::Point::default();
        let mut max_loc = core::Point::default();
        min_max_loc(&flow, Some(&mut min_val), Some(&mut max_val), 
                    Some(&mut min_loc), Some(&mut max_loc), &core::no_array())?;
        
        info!("Flow map range: [{}, {}], min loc: [{}, {}], processing time: {:?}", 
                min_val, max_val, min_loc.x, min_loc.y, duration);
              
        if (max_val - min_val).abs() < std::f64::EPSILON {
            warn!("Flow map is flat - possible error in flood fill!");
        }

        // the min loc should the same as one of the start locations
        let min_loc = (min_loc.x, min_loc.y);   
        if !locations.contains(&min_loc) {
            warn!("Min location {:?} is not one of the start locations: {:?}", min_loc, locations);
        }
        
        self.flow = Some(flow);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use opencv::core::Scalar;

    use super::*;

    fn create_test_image() -> Mat {
        Mat::new_rows_cols_with_default(100, 100, opencv::core::CV_8UC3, Scalar::all(255.0)).unwrap()
    }

    #[test]
    fn test_new() {
        let image = create_test_image();
        let processed_image = ProcessedImage::new(image.clone(), 8, false);
        assert_eq!(processed_image.image().size().unwrap(), image.size().unwrap());
        assert_eq!(processed_image.poster_colors, 8);
        assert_eq!(processed_image.convert_to_lab, false);
    }

    #[test]
    fn test_set_image() {
        let image = create_test_image();
        let mut processed_image = ProcessedImage::new(image.clone(), 8, false);
        let new_image = create_test_image();
        processed_image.set_image(new_image.clone());
        assert_eq!(processed_image.image().size().unwrap(), new_image.size().unwrap());
        assert!(processed_image.resized_image.is_none());
        assert!(processed_image.posterized_image.is_none());
        assert!(processed_image.flow.is_none());
    }

    #[test]
    fn test_resize() {
        let image = create_test_image();
        let mut processed_image = ProcessedImage::new(image.clone(), 8, false);
        let new_size = Size::new(50, 50);
        processed_image.resize(Some(new_size)).unwrap();
        assert_eq!(processed_image.resized_image.as_ref().unwrap().size().unwrap(), new_size);
    }

    #[test]
    fn test_posterize() {
        let image = create_test_image();
        let mut processed_image = ProcessedImage::new(image.clone(), 8, false);
        processed_image.resize(None).unwrap();
        processed_image.posterize().unwrap();
        assert!(processed_image.posterized_image.is_some());
    }

    #[test]
    fn test_flow_map() {
        let image = create_test_image();
        let mut processed_image = ProcessedImage::new(image.clone(), 8, false);
        processed_image.resize(None).unwrap();
        processed_image.posterize().unwrap();
        processed_image.flow_map(None).unwrap();
        assert!(processed_image.flow.is_some());
    }

    #[test]
    fn test_process() {
        let image = create_test_image();
        let mut processed_image = ProcessedImage::new(image.clone(), 8, false);
        let new_size = Size::new(50, 50);
        processed_image.process(Some(new_size), None).unwrap();
        assert_eq!(processed_image.resized_image.as_ref().unwrap().size().unwrap(), new_size);
        assert!(processed_image.posterized_image.is_some());
        assert!(processed_image.flow.is_some());
    }
}
