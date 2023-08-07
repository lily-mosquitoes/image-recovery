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

//! Implementation of algorithms for image recovery.
use std::ops::Deref;

use ndarray::{
    Array3,
    ShapeError,
};

use crate::{
    image_array::ImageArray,
    ops::{
        Average,
        Gradient,
        Norm,
        VectorLen,
    },
};

impl ImageArray<Array3<f64>> {
    /// Image denoising algorithm for 2 dimentional shapes with 1 dimention of
    /// information (pixels) as an arbitrarily sized vector. Assumes axes 0
    /// and 1 and the x and y coordinates of the image, and axis 2 is the
    /// pixel vector coordinate of the image.
    ///
    /// # inputs
    /// `lambda` is the target value of the dual objective function,
    /// i.e. how close you want the output to be to the input:
    /// approaching 0, the output should be completely smooth (flat),
    /// approaching "infinifty", the output should be the same as
    /// the original input.
    ///
    /// `tau` and `sigma` affect how fast the algorithm converges,
    /// according to Chambolle, A. and Pock, T. (2011) these should
    /// be chosen such that `tau * lambda * L2 norm^2 <= 1` where
    /// `L2 norm^2 <= 8`.
    ///
    /// `gamma` updates the algorithm's internal variables,
    /// for the accelerated algorithm of Chambolle, A. and Pock, T. (2011)
    /// the chosen value is `0.35 * lambda`.
    ///
    /// `max_iter` and `convergence_threshold` bound the runtime of the
    /// algorithm, i.e. it runs until `convergence_threshold < norm(current -
    /// previous) / norm(previous)` or `max_iter` is hit.
    pub fn denoise(
        &self,
        lambda: f64,
        mut tau: f64,
        mut sigma: f64,
        gamma: f64,
        max_iter: u32,
        convergence_threshold: f64,
    ) -> Result<Self, ShapeError> {
        // primal variable (two copies, for storing value of iteration n-1)
        let mut current: Array3<f64> = self.deref().clone();
        let mut previous: Array3<f64>;
        // primal variable "bar"
        let mut current_bar = current.clone();
        // dual variables
        let mut dual_a = current.positive_gradient_on_axis(0)?;
        let mut dual_b = current.positive_gradient_on_axis(1)?;
        // theta will be set upon first iteration
        let mut theta: f64;

        let mut iter: u32 = 1;
        loop {
            // update the dual variable
            dual_a =
                &dual_a + (sigma * current_bar.positive_gradient_on_axis(0)?);
            dual_b =
                &dual_b + (sigma * current_bar.positive_gradient_on_axis(1)?);
            // project dual variables color axis into L2 ball (-1, 1).
            // assumes axis 2 is color axis of image.
            let max = dual_a
                .vector_len_on_axis(&dual_b, 2)?
                .map(|&x| 1_f64.max(x));
            dual_a /= &max;
            dual_b /= &max;

            // update the primal variable
            previous = current.clone();
            current = &current
                - (tau
                    * (dual_a.negative_gradient_on_axis(0)?
                        + dual_b.negative_gradient_on_axis(1)?));
            current = self.weighted_average(&current, tau, lambda);

            // update theta
            theta = 1_f64 / (1_f64 + (2_f64 * gamma * tau));
            // update tau
            tau *= theta;
            // update sigma
            sigma /= theta;

            // update the primal variable bar
            current_bar = &current + &(theta * (&current - &previous));

            // check for convergence or max_iter iterations
            let c = (&current - &previous).norm() / previous.norm();
            if c < convergence_threshold || iter >= max_iter {
                log::debug!(
                    "returned at iteration = {}; where max = {}",
                    iter,
                    max_iter
                );
                log::debug!(
                    "convergence = {}; where threshold = {}",
                    c,
                    convergence_threshold
                );
                break;
            }
            iter += 1;
        }

        Ok(ImageArray::from(&current))
    }
}
