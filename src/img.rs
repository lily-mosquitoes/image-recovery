//! Struct and Traits for loading RGB images (`image::RgbImage`) into a set of 3 matrices (`RbgMatrices`) representing each color channel (Red, Green and Blue) as a matrix (`ndarray::Array2<f64>`), and vice-versa.

use image::RgbImage;
use crate::RgbMatrices;

pub trait Shape {
    fn width(&self) -> usize;
    fn height(&self) -> usize;
}

#[allow(dead_code)]
#[derive(Clone, Copy)]
#[repr(usize)]
pub enum Channel {
    Red = 0,
    Green = 1,
    Blue = 2,
}

pub trait Manipulation {
    fn shape(&self) -> (usize, usize);
    fn to_matrices(&self) -> RgbMatrices;
    fn from_matrices(img_matrices: &RgbMatrices) -> RgbImage;
}
