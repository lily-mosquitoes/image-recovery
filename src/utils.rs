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

//! Utility functions for matrices (`ndarray::Array2<f64>`).

use ndarray::Array2;
use crate::{
    RgbMatrices,
    array_ops::{Power},
};

/// length of vectors given two matrices, unchecked for size.
pub fn len_of_vectors(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    (a.squared() + b.squared())
        .map(|x| x.sqrt())
}

/// length of vectors given two RGB matrices, unchecked for size.
///
/// This modification is inspired by the work of Bredies, K. (2014).
pub fn len_of_vectors_multichannel(a: &RgbMatrices, b: &RgbMatrices) -> Array2<f64> {
    let l = a.squared() + b.squared();

    (l.red + l.green + l.blue).map(|x| x.sqrt())
}

// previously used this implementation, which incidentally
// this encourages sharp edges for the output!
// pub fn len_of_vectors_multichannel(a: &RgbMatrices, b: &RgbMatrices) -> Array2<f64> {
//     let l = (a.squared() + b.squared())
//         .map(|x| x.sqrt());
//
//     l.red + l.green + l.blue
// }

/// the projection of vectors from two matrices into a 2D ball (-1, 1), unchecked for size.
pub fn ball_projection(a: &Array2<f64>, b: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
    let max = len_of_vectors(a, b)
        .map(|x| 1_f64.max(*x));

    (a / &max, b / &max)
}

/// the projection of vectors from two RGB matrices into a 2D ball (-1, 1), unchecked for size.
pub fn ball_projection_multichannel(a: &RgbMatrices, b: &RgbMatrices) -> (RgbMatrices, RgbMatrices) {
    let max = len_of_vectors_multichannel(a, b)
        .map(|x| 1_f64.max(*x));

    (a / &max, b / &max)
}
