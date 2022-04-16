//! Traits for operations on matrices (`ndarray::Array2<f64>`)

use ndarray::Array2;

pub trait Derivative {
    // Derivative on the X axis (wrapping)
    fn dx(&self) -> Self;
    // Derivative on the X axis, transposed (wrapping)
    fn dx_transposed(&self) -> Self;
    // Derivative on the Y axis (wrapping)
    fn dy(&self) -> Self;
    // Derivative on the Y axis, transposed (wrapping)
    fn dy_transposed(&self) -> Self;
}

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

pub trait Power {
    // power of 2
    fn squared(&self) -> Self;
    // power of n, where n is an unsigned 32-bit integer
    fn powi(&self, n: i32) -> Self;
    // power of n, where n is a 64-bit floating point number
    fn powf(&self, n: f64) -> Self;
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
