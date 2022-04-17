//! Utility functions for matrices (`ndarray::Array2<f64>`)

use ndarray::Array2;
use crate::{
    RgbMatrices,
    array_ops::{Power},
};

// length of vectors given two matrices, unchecked for size
pub fn len_of_vectors(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    (a.squared() + b.squared())
        .map(|x| x.sqrt())
}

// length of vectors given two RGB matrices, unchecked for size
pub fn len_of_vectors_multichannel(a: &RgbMatrices, b: &RgbMatrices) -> Array2<f64> {
    let l = a.squared() + b.squared();

    (l.red + l.green + l.blue).map(|x| x.sqrt())
}

// this encourages sharp edges
// pub fn len_of_vectors_multichannel(a: &RgbMatrices, b: &RgbMatrices) -> Array2<f64> {
//     let l = (a.squared() + b.squared())
//         .map(|x| x.sqrt());
//
//     l.red + l.green + l.blue
// }

// the projection of vectors from two matrices into a 2D ball (-1, 1),
// unchecked for size
pub fn ball_projection(a: &Array2<f64>, b: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
    let max = len_of_vectors(a, b)
        .map(|x| 1_f64.max(*x));

    (a / &max, b / &max)
}

// the projection of vectors from two RGB matrices into a 2D ball (-1, 1),
// unchecked for size
pub fn ball_projection_multichannel(a: &RgbMatrices, b: &RgbMatrices) -> (RgbMatrices, RgbMatrices) {
    let max = len_of_vectors_multichannel(a, b)
        .map(|x| 1_f64.max(*x));

    (a / &max, b / &max)
}
