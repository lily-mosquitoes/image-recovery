//! Traits for operations on matrices (`ndarray::Array2<f64>`)

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

pub trait Power {
    // power of 2
    fn squared(&self) -> Self;
    // power of n, where n is an unsigned 32-bit integer
    fn powi(&self, n: i32) -> Self;
    // power of n, where n is a 64-bit floating point number
    fn powf(&self, n: f64) -> Self;
}
