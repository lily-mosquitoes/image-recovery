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
//!     ImageArray, // struct for holding images
//! };
//!
//! fn main() {
//!     // the `image` crate provides functionality to decode images
//!     let img = image::open("examples/source_images/angry_birb_noisy.png")
//!         .expect("image could not be open")
//!         .into_rgb8(); // the algorithms in this library are implemented for the Luma and Rgb types
//!     // transform the RGB image into a 3D Array
//!     let img_array = ImageArray::from(&img);
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
//!     let max_iter: u32 = 100;
//!     // the algorithm will stop running if:
//!     // `convergence_threshold < norm(current - previous) / norm(previous)`
//!     // where `current` is the output candidate for the current iteration,
//!     // and `previous` is the output candidate of the previous iteration.
//!     let convergence_threshold = 10_f64.powi(-10);
//!
//!     // now we can call the denoising solver with the chosen variables
//!     let denoised_array = img_array
//!         .denoise(lambda, tau, sigma, gamma, max_iter, convergence_threshold)
//!         .unwrap(); // will fail if image shape is 1 pixel in either x or y
//!
//!     // we convert the solution into an RGB image format
//!     let denoised_img = denoised_array.into_rgb();
//!
//!     // encode it and save it to a file
//!     denoised_img.save("examples/result_images/angry_birb_denoised.png")
//!         .expect("image could not be saved");
//! }
//! ```
//!
//! This should provide the following result:
//!
//! Source image: | Output image:
//! ---|---
//! ![source image, noisy](https://github.com/lily-mosquitoes/image-recovery/raw/main/examples/source_images/angry_birb_noisy.png) | ![output image, denoised](https://github.com/lily-mosquitoes/image-recovery/raw/main/examples/result_images/angry_birb_denoised.png)

#![feature(test)]
extern crate test;

mod image_array;
mod ops;
mod solvers;

pub use image;
pub use image_array::ImageArray;
pub use ndarray;
