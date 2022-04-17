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
