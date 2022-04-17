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
