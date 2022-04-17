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

//! Implementation for operations on matrices (`ndarray::Array2<f64>`)

use ndarray::Array2;
use crate::{
    array_ops::{Derivative, Power},
    RgbMatrices,
};

impl Derivative for Array2<f64> {
    // Derivative on the X axis (wrapping)
    fn dx(&self) -> Self {
        // shift to the right (wrapping)
        let width = self.ncols();
        let last_col = self.slice(ndarray::s![.., width-1..]);
        let remaining_cols = self.slice(ndarray::s![.., ..width-1]);
        let rshift_matrix = ndarray::concatenate(ndarray::Axis(1), &[last_col, remaining_cols]).unwrap();

        self - &rshift_matrix
    }

    // Derivative on the X axis, transposed (wrapping)
    fn dx_transposed(&self) -> Self {
        // shift to the left (wrapping)
        let first_col = self.slice(ndarray::s![.., ..1]);
        let remaining_cols = self.slice(ndarray::s![.., 1..]);
        let lshift_matrix = ndarray::concatenate(ndarray::Axis(1), &[remaining_cols, first_col]).unwrap();

        self - &lshift_matrix
    }

    // Derivative on the Y axis (wrapping)
    fn dy(&self) -> Self {
        // shift it down (wrapping)
        let height = self.nrows();
        let last_row = self.slice(ndarray::s![height-1.., ..]);
        let remaining_rows = self.slice(ndarray::s![..height-1, ..]);
        let dshift_matrix = ndarray::concatenate(ndarray::Axis(0), &[last_row, remaining_rows]).unwrap();

        self - &dshift_matrix
    }

    // Derivative on the Y axis, transposed (wrapping)
    fn dy_transposed(&self) -> Self {
        // shift it up (wrapping)
        let first_row = self.slice(ndarray::s![..1, ..]);
        let remaining_rows = self.slice(ndarray::s![1.., ..]);
        let ushift_matrix = ndarray::concatenate(ndarray::Axis(0), &[remaining_rows, first_row]).unwrap();

        self - &ushift_matrix
    }
}

impl Derivative for RgbMatrices {
    // Derivative on the X axis (wrapping)
    fn dx(&self) -> Self {
        RgbMatrices {
            shape: self.shape,
            red: self.red.dx(),
            green: self.green.dx(),
            blue: self.blue.dx(),
        }
    }

    // Derivative on the X axis, transposed (wrapping)
    fn dx_transposed(&self) -> Self {
        RgbMatrices {
            shape: self.shape,
            red: self.red.dx_transposed(),
            green: self.green.dx_transposed(),
            blue: self.blue.dx_transposed(),
        }
    }

    // Derivative on the Y axis (wrapping)
    fn dy(&self) -> Self {
        RgbMatrices {
            shape: self.shape,
            red: self.red.dy(),
            green: self.green.dy(),
            blue: self.blue.dy(),
        }
    }

    // Derivative on the Y axis, transposed (wrapping)
    fn dy_transposed(&self) -> Self {
        RgbMatrices {
            shape: self.shape,
            red: self.red.dy_transposed(),
            green: self.green.dy_transposed(),
            blue: self.blue.dy_transposed(),
        }
    }
}

impl Power for Array2<f64> {
    // element-wise power of 2
    fn squared(&self) -> Self {
        self * self
    }

    // element-wise power of i, where i is an unsigned 32-bit integer
    fn powi(&self, n: i32) -> Self {
        self.map(|x| x.powi(n))
    }

    // element-wise power of i, where i is 64-bit floating point number
    fn powf(&self, n: f64) -> Self {
        self.map(|x| x.powf(n))
    }
}

impl Power for RgbMatrices {
    // element-wise power of 2
    fn squared(&self) -> Self {
        self * self
    }

    // element-wise power of i, where i is an unsigned 32-bit integer
    fn powi(&self, n: i32) -> Self {
        RgbMatrices {
            shape: self.shape,
            red: self.red.map(|x| x.powi(n)),
            green: self.green.map(|x| x.powi(n)),
            blue: self.blue.map(|x| x.powi(n)),
        }
    }

    // element-wise power of i, where i is 64-bit floating point number
    fn powf(&self, n: f64) -> Self {
        RgbMatrices {
            shape: self.shape,
            red: self.red.map(|x| x.powf(n)),
            green: self.green.map(|x| x.powf(n)),
            blue: self.blue.map(|x| x.powf(n)),
        }
    }
}
