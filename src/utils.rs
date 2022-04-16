//! Utility functions for matrices (ndarray::Array2<f64>)

use ndarray::Array2;
use crate::ops::{Power};

// length of vectors given two matrices, unchecked for size
pub fn len_of_vectors(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    (a.squared() + b.squared()).map(|x| x.sqrt())
}

// let len_of_gradient= |a, b| { len_of_vectors(a.dx(), a.dy()) };

// the projection of vectors from two matrices into a 2D ball (-1, 1),
// unchecked for size
pub fn ball_projection(a: &Array2<f64>, b: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
    let max = len_of_vectors(a, b).map(|x| 1_f64.max(*x));

    (a / &max, b / &max)
}
