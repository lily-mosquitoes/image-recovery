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

//! Image-recovery algorithms implemented in Rust.

#![feature(test)]
extern crate test;

pub mod array_ops;
pub mod img;
pub mod utils;
pub mod solvers;
// #[cfg(test)] mod solvers_tests;
mod _impl;
#[cfg(test)] mod tests;

use ndarray::Array2;

#[derive(Debug, Clone, PartialEq)]
pub struct RgbMatrices {
    pub shape: (usize, usize),
    pub red: Array2<f64>,
    pub green: Array2<f64>,
    pub blue: Array2<f64>,
}

impl RgbMatrices {
    #[inline]
    #[must_use]
    pub fn new(shape: (usize, usize)) -> Self {
        RgbMatrices {
            shape,
            red: Array2::<f64>::zeros(shape),
            green: Array2::<f64>::zeros(shape),
            blue: Array2::<f64>::zeros(shape),
        }
    }

    #[must_use]
    pub fn from_channels(red: &Array2<f64>, green: &Array2<f64>, blue: &Array2<f64>) -> Self {
        let red_shape = (red.ncols(), red.nrows());
        let green_shape = (green.ncols(), green.nrows());
        let blue_shape = (blue.ncols(), blue.nrows());

        if !((red_shape == green_shape) && (red_shape == blue_shape)) {
            panic!("arrays must be all of the same shape");
        }

        RgbMatrices {
            shape: red_shape,
            red: red.to_owned(),
            green: green.to_owned(),
            blue: blue.to_owned(),
        }
    }

    pub fn sum(&self) -> f64 {
        (&self.red + &self.green + &self.blue).sum()
    }

    pub fn map<F>(&self, mut closure: F) -> RgbMatrices
    where
        F: FnMut(&f64) -> f64,
    {
        RgbMatrices {
            shape: self.shape,
            red: self.red.map(&mut closure),
            green: self.green.map(&mut closure),
            blue: self.blue.map(&mut closure),
        }
    }
}
