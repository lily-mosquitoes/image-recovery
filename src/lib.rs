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

//! # Image recovery algorithms, implemented in Rust.
//!
//! The solvers on this library are based on the algorithms presented in [Chambolle, A. and Pock, T. (2011)](https://link.springer.com/article/10.1007/s10851-010-0251-1), with modifications inspired by [Bredies, K. (2014)](https://link.springer.com/chapter/10.1007/978-3-642-54774-4_3).
//!
//! Uses the [`image` crate](https://docs.rs/image/latest/image/) for loading and saving images, and the [`ndarray` crate](https://docs.rs/ndarray/latest/ndarray/index.html) for manipulating matrices.
//!
//! # How to use it:
//! Declare the dependency in you Cargo.toml
//!
//! ```toml
//! [dependencies]
//! image-recovery = "0.1"
//! ```
//!
//! # Examples:
//!
//! Examples for each solver can be found in the [`examples` folder](https://github.com/lily-mosquitoes/image-recovery/examples), and those can be run with `cargo run --example example_name`. However, a quick example usage is shown below:
//!
//! ## Image denoising (multichannel)
//!
//! ```rust
//! use image_recovery::{
//!     image, // re-exported `image` crate
//!     img::Manipulation, // trait for image::RgbImage manipulation
//!     solvers, // module with image recovery algorithms
//! };
//!
//! fn main() {
//!     // the `image` crate provides functionality to decode images
//!     let img = image::open("examples/source_images/angry_birb_noisy.png")
//!         .expect("image could not be open")
//!         .into_rgb8(); // the algorithms in this library are implemented for RGB images
//!
//!     // load the RGB image into an object which is composed
//!     // of 3 matrices, one for each channel
//!     let img_matrices = img.to_matrices();
//!
//!     // choose inputs for the denoising solver:
//!     // according to Chambolle, A. and Pock, T. (2011),
//!     // tau and lambda should be chosen such that
//!     // `tau * lambda * L2 norm^2 <= 1`
//!     // while `L2 norm^2 <= 8`
//!     // If we choose `tau * lambda * L2 norm^2 == 1`, then:
//!     let tau: f64 = 1.0 / 2_f64.sqrt();
//!     let sigma: f64 = 1_f64 / (8.0 * tau);
//!
//!     // lambda drives the dual objective function
//!     // closer to zero results in a smoother output image
//!     // closer to infinity results in an output closer to the input
//!     let lambda: f64 = 0.0259624705;
//!
//!     // gamma is a variable used to update the internal
//!     // state of the algorithm's variables, providing
//!     // an accelerated method for convergence.
//!     // Chambolle, A. and Pock, T. (2011), choose
//!     // the value to be `0.35 * lambda`
//!     let gamma: f64 = 0.35 * lambda;
//!
//!     // choose bounds for denoising solver
//!     // the algorithm will run for at most `max_iter` iterations
//!     let max_iter: u32 = 500;
//!     // the algorithm will stop running if:
//!     // `convergence_threshold < norm(current - previous) / norm(previous)`
//!     // where `current` is the output candidate for the current iteration,
//!     // and `previous` is the output candidate of the previous iteration.
//!     let convergence_threshold = 10_f64.powi(-10);
//!
//!     // now we can call the denoising solver with the chosen variables
//!     let denoised = solvers::denoise_multichannel(&img_matrices, lambda, tau, sigma, gamma, max_iter, convergence_threshold);
//!
//!     // we convert the solution into an RGB image format
//!     let new_img = image::RgbImage::from_matrices(&denoised);
//!
//!     // encode it and save it to a file
//!     new_img.save("examples/result_images/angry_birb_denoised_multichannel.png")
//!         .expect("image could not be saved");
//! }
//! ```
//!
//! This should provide the following result:
//!
//! Source image: | Output image:
//! ---|---
//! ![source image, noisy](https://github.com/lily-mosquitoes/image-recovery/raw/main/examples/source_images/angry_birb_noisy.png) | ![output image, denoised](https://github.com/lily-mosquitoes/image-recovery/raw/main/examples/result_images/angry_birb_denoised_multichannel.png)

#![feature(test)]
extern crate test;

mod _impl;
pub mod array_ops;
pub mod img;
pub mod solvers;
#[cfg(test)]
mod tests;
pub mod utils;

pub use image;
pub use ndarray::Array2;

/// struct representing an RGB image as 3 matrices,
/// one for each color channel
#[derive(Debug, Clone, PartialEq)]
pub struct RgbMatrices {
    pub shape: (usize, usize),
    pub red: Array2<f64>,
    pub green: Array2<f64>,
    pub blue: Array2<f64>,
}

impl RgbMatrices {
    /// generates a new `RgbMatrices` with all channels initialized to matrices of the given `shape`, but full of zeroes.
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

    // generates a new `RgbMatrices` from the given 3 matrices, each representing a color channel. Panics if their shapes aren't all the same.
    #[must_use]
    pub fn from_channels(red: &Array2<f64>, green: &Array2<f64>, blue: &Array2<f64>) -> Self {
        let red_shape = red.raw_dim();
        let green_shape = green.raw_dim();
        let blue_shape = blue.raw_dim();

        if !((red_shape == green_shape) && (red_shape == blue_shape)) {
            panic!("arrays must be all of the same shape");
        }

        RgbMatrices {
            shape: (red_shape[0], red_shape[1]),
            red: red.to_owned(),
            green: green.to_owned(),
            blue: blue.to_owned(),
        }
    }

    /// returns the sum of all elements in the `RgbMatrices`.
    pub fn sum(&self) -> f64 {
        (&self.red + &self.green + &self.blue).sum()
    }

    /// applies the given closure by calling `map(closure)` on each channel of the `RgbMatrices`. Useful for element-wise operations.
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
