// Copyright (C) 2022  Lílian Ferreira de Freitas & Emilia L. K. Blåsten
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published
// by the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

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

    fn from_matrices(img_matrices: &RgbMatrices) -> Self {
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
