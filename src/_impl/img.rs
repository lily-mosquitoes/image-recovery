//! Implementation of Traits for loading RGB images (`image::RgbImage`) into a set of 3 matrices (`RbgMatrices`) representing each color channel (Red, Green and Blue) as a matrix (`ndarray::Array2<f64>`), and vice-versa.

use image::{RgbImage, Rgb};
use crate::{
    RgbMatrices,
    img::{Channel, Shape, Manipulation},
};

impl Shape for (usize, usize) {
    fn width(&self) -> usize {
        self.0
    }

    fn height(&self) -> usize {
        self.1
    }
}

impl Manipulation for RgbImage {
    fn shape(&self) -> (usize, usize) {
        let (width, height) = self.dimensions();

        (width as usize, height as usize)
    }

    fn to_matrices(&self) -> RgbMatrices {
        let shape = self.shape();

        // initialize matrices (full of zeroes)
        // of the same shape (width and height) of the RgbImage
        let mut img_matrices = RgbMatrices::new(shape);

        // iterate through every pixel
        // get the the values for each channel of the pixel
        // and put it inside each channel matrix,
        // at the same x, y position
        for x in 0..shape.width() {
            for y in 0..shape.height() {
                let pixel: &Rgb<u8> = self.get_pixel(x as u32, y as u32);
                img_matrices.red[[x, y]] = pixel[Channel::Red as usize] as f64;
                img_matrices.green[[x, y]] = pixel[Channel::Green as usize] as f64;
                img_matrices.blue[[x, y]] = pixel[Channel::Blue as usize] as f64;
            }
        }

        img_matrices
    }

    fn from_matrices(img_matrices: &RgbMatrices) -> RgbImage {
        let shape = img_matrices.shape;

        // initialize image (full of zeroes)
        // of the same shape (width and height) of the RgbMatrices
        let mut img = RgbImage::new(shape.width() as u32, shape.height() as u32);

        // iterate through every pixel
        // get the values for each channel from the matrices
        // and put it inside the pixel at the channel's location,
        // at the same x, y position
        for x in 0..shape.width() {
            for y in 0..shape.height() {
                let pixel: &mut Rgb<u8> = img.get_pixel_mut(x as u32, y as u32);
                pixel[Channel::Red as usize] = img_matrices.red[[x, y]] as u8;
                pixel[Channel::Green as usize] = img_matrices.green[[x, y]] as u8;
                pixel[Channel::Blue as usize] = img_matrices.blue[[x, y]] as u8;
            }
        }

        img
    }
}
