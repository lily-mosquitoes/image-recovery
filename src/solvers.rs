use ndarray::Array2;
use crate::{
    RgbMatrices,
    array_ops::{Derivative, Power},
    utils,
};

// J Math Imaging Vis (2011) 40: 120–145
// DOI 10.1007/s10851-010-0251-1

// lambda => the target value of the dual objective function,
// i.e. how close you want to be to the original matrix (image):
// 0 == completely flat; INFINITY == same as original.
//
// tau, sigma => meaning?
//
// norm_squared => affects how fast the algorithm converges,
// should be at most 8.0.
//
// gamma => ???
pub fn denoise(input: &Array2<f64>, lambda: f64, mut tau: f64, mut sigma: f64, norm_squared: f64, gamma: f64) -> Array2<f64> {
    if tau * sigma * norm_squared > 1_f64 {
        panic!("must satisfy `tau * sigma * norm_squared <= 1`")
    }

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

    let mut iter = 1;
    loop {
        // update the dual variable
        (dual_a, dual_b) = utils
            ::ball_projection(&f(dual_a, current_bar.dx(), sigma), &f(dual_b, current_bar.dy(), sigma));

        // update the primal variable
        // save it first
        previous = current.to_owned();
        current = weighted_average(&current - (tau *
            k_star(&dual_a, &dual_b)),
            tau,
            lambda);

        // update theta
        theta = 1_f64 / (1_f64 + (2_f64 * gamma * tau));
        // update tau
        tau = theta * tau;
        // update sigma
        sigma = sigma / theta;

        // update the primal variable bar
        current_bar = &current + &((theta * (&current - &previous)));

        // check for convergence or 500 iterations
        let c = norm(&(&current - &previous)) / norm(&previous);
        if c < 10_f64.powi(-10) || iter >= 500 {
            println!("returned at iter n {}", iter);
            break
        }
        iter += 1;
    }

    current
}

pub fn denoise_multichannel(input: &RgbMatrices, lambda: f64, mut tau: f64, mut sigma: f64, norm_squared: f64, gamma: f64) -> RgbMatrices {
    if tau * sigma * norm_squared > 1_f64 {
        panic!("must satisfy `tau * sigma * norm_squared <= 1`")
    }

    // primal variable (two copies, for storing value of iteration n-1)
    let mut current = input.clone();
    let mut previous: RgbMatrices;
    // primal variable "bar"
    let mut current_bar = current.clone();
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

    let mut iter = 1;
    loop {
        // update the dual variable
        (dual_a, dual_b) = utils
            ::ball_projection_multichannel(&f(dual_a, current_bar.dx(), sigma), &f(dual_b, current_bar.dy(), sigma));

        // update the primal variable
        // save it first
        previous = current.clone();
        current = weighted_average(&current - (tau *
            k_star(&dual_a, &dual_b)),
            tau,
            lambda);

        // update theta
        theta = 1_f64 / (1_f64 + (2_f64 * gamma * tau));
        // update tau
        tau = theta * tau;
        // update sigma
        sigma = sigma / theta;

        // update the primal variable bar
        current_bar = &current + &((theta * (&current - &previous)));

        // check for convergence or 500 iterations
        let c = norm(&(&current - &previous)) / norm(&previous);
        if c < 10_f64.powi(-10) || iter >= 500 {
            println!("returned at iter n {}", iter);
            break
        }
        iter += 1;
    }

    current
}
