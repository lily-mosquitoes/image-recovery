//! Trait for getting single channel matrices (`ndarray::Array2<f64>`) from RGB images (image::RgbImage) and updating RGB images from single channel matrices

use ndarray::Array2;
use image::{RgbImage, Rgb};

#[allow(dead_code)]
#[derive(Clone, Copy)]
#[repr(usize)]
pub enum Channel {
    Red = 0,
    Green = 1,
    Blue = 2,
}

#[derive(Debug, Clone)]
pub struct ImageDimentionError;

impl std::fmt::Display for ImageDimentionError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "input array has invalid dimensions")
    }
}

pub trait Manipulation {
    fn get_channel(&self, channel: Channel) -> Array2<f64>;
    fn update_channel(&mut self, channel: Channel, matrix: &Array2<f64>) -> Result<(), ImageDimentionError>;
}

impl Manipulation for RgbImage {
    fn get_channel(&self, channel: Channel) -> Array2<f64> {
        // the function Rgb.dimensions() returns a u32 tuple
        let dimensions: (u32, u32) = self.dimensions();
        // we type cast it to a usize tuple for ease of work with the
        // ndarray Array2 functions
        let dimensions: (usize, usize) = (dimensions.0 as usize, dimensions.1 as usize);

        // initialize single channel matrix (full of zeroes)
        // of the same dimensions of the RgbImage
        let mut matrix = Array2::<f64>::zeros(dimensions);

        // iterate through every pixel
        // get the the given Channel value at the pixel
        // and put it inside the matrix at the same x, y position
        for x in 0..dimensions.0 {
            for y in 0..dimensions.1 {
                let pixel: &Rgb<u8> = self.get_pixel(x as u32, y as u32);
                matrix[[x, y]] = pixel[channel as usize] as f64;
            }
        }

        // return single channel matrix
        matrix
    }

    fn update_channel(&mut self, channel: Channel, matrix: &Array2<f64>) -> Result<(), ImageDimentionError> {
        // the function Rgb.dimensions() returns a u32 tuple
        let dimensions: (u32, u32) = self.dimensions();

        // make sure the given array is of the same size as the RgbImage
        if (matrix.nrows() as u32, matrix.ncols() as u32) != dimensions {
            return Err(ImageDimentionError);
        }

        // iterate through every item of the array (single channel "pixel")
        // get mut ref to the Rgb pixel at x, y from the RgbImage
        // put the matrix value at the correct position for the chosen Channel in this pixel
        for x in 0..dimensions.0 {
            for y in 0..dimensions.1 {
                let pixel_array: &mut Rgb<u8> = self.get_pixel_mut(x, y);
                // Array2 is indexed with usize
                pixel_array[channel as usize] = matrix[[x as usize, y as usize]] as u8;
            }
        }

        Ok(())
    }
}
