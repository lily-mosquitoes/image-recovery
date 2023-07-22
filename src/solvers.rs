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

use crate::{
    array_ops::{Derivative, Power},
    utils, RgbMatrices,
};
use ndarray::Array2;

/// single channel denoising algorithm.
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
/// algorithm, i.e. it runs until `convergence_threshold < norm(current - previous) / norm(previous)` or `max_iter` is hit.
pub fn denoise(
    input: &Array2<f64>,
    lambda: f64,
    mut tau: f64,
    mut sigma: f64,
    gamma: f64,
    max_iter: u32,
    convergence_threshold: f64,
) -> Array2<f64> {
    // primal variable (two copies, for storing value of iteration n-1)
    let mut current = input.to_owned();
    let mut previous: Array2<f64>;
    // primal variable "bar"
    let mut current_bar = current.to_owned();
    // dual variable
    let (mut dual_a, mut dual_b) = (current.dx(), current.dy());
    // theta will be set upon first iteration
    let mut theta: f64;

    // helper function for dual update
    let f = |dual, gradient, sigma| dual + (sigma * gradient);

    // helper function for convergence
    let norm = |m: &Array2<f64>| m.squared().sum().sqrt();

    // helper functions for primal update
    // this calculates a weighted average between the original noisy image (input) and the given image to this operator (u)
    let weighted_average = |u, t, l| (u + (t * l * input)) / (1_f64 + t * l);
    // divergence function
    let k_star = |a: &Array2<f64>, b: &Array2<f64>| a.dx_transposed() + b.dy_transposed();

    let mut iter: u32 = 1;
    loop {
        // update the dual variable
        (dual_a, dual_b) = utils::ball_projection(
            &f(dual_a, current_bar.dx(), sigma),
            &f(dual_b, current_bar.dy(), sigma),
        );

        // update the primal variable
        // save it first
        previous = current.to_owned();
        current = weighted_average(&current - (tau * k_star(&dual_a, &dual_b)), tau, lambda);

        // update theta
        theta = 1_f64 / (1_f64 + (2_f64 * gamma * tau));
        // update tau
        tau *= theta;
        // update sigma
        sigma /= theta;

        // update the primal variable bar
        current_bar = &current + &(theta * (&current - &previous));

        // check for convergence or max_iter iterations
        let c = norm(&(&current - &previous)) / norm(&previous);
        if c < convergence_threshold || iter >= max_iter {
            log::debug!("returned at iter n {}", iter);
            break;
        }
        iter += 1;
    }

    current
}

/// multichannel denoising algorithm.
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
/// algorithm, i.e. it runs until `convergence_threshold < norm(current - previous) / norm(previous)` or `max_iter` is hit.
pub fn denoise_multichannel(
    input: &RgbMatrices,
    lambda: f64,
    mut tau: f64,
    mut sigma: f64,
    gamma: f64,
    max_iter: u32,
    convergence_threshold: f64,
) -> RgbMatrices {
    // primal variable (two copies, for storing value of iteration n-1)
    let mut current = input.to_owned();
    let mut previous: RgbMatrices;
    // primal variable "bar"
    let mut current_bar = current.to_owned();
    // dual variable
    let (mut dual_a, mut dual_b) = (current.dx(), current.dy());
    // theta will be set upon first iteration
    let mut theta: f64;

    // helper function for dual update
    let f = |dual, gradient, sigma| dual + (sigma * gradient);

    // helper function for convergence
    let norm = |m: &RgbMatrices| m.squared().sum().sqrt();

    // helper functions for primal update
    // this calculates a weighted average between the original noisy image (input) and the given image to this operator (u)
    let weighted_average = |u, t, l| (u + (t * l * input)) / (1_f64 + t * l);
    // divergence function
    let k_star = |a: &RgbMatrices, b: &RgbMatrices| a.dx_transposed() + b.dy_transposed();

    let mut iter: u32 = 1;
    loop {
        // update the dual variable
        (dual_a, dual_b) = utils::ball_projection_multichannel(
            &f(dual_a, current_bar.dx(), sigma),
            &f(dual_b, current_bar.dy(), sigma),
        );

        // update the primal variable
        // save it first
        previous = current.to_owned();
        current = weighted_average(&current - (tau * k_star(&dual_a, &dual_b)), tau, lambda);

        // update theta
        theta = 1_f64 / (1_f64 + (2_f64 * gamma * tau));
        // update tau
        tau *= theta;
        // update sigma
        sigma /= theta;

        // update the primal variable bar
        current_bar = &current + &(theta * (&current - &previous));

        // check for convergence or max_iter iterations
        let c = norm(&(&current - &previous)) / norm(&previous);
        if c < convergence_threshold || iter >= max_iter {
            log::debug!("returned at iter n {}", iter);
            break;
        }
        iter += 1;
    }

    current
}
