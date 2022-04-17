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

//! Traits for operations on matrices (`ndarray::Array2<f64>`).

/// trait functions for differentiation
pub trait Derivative {
    /// Derivative on the X axis (wrapping)
    fn dx(&self) -> Self;
    /// Derivative on the X axis, transposed (wrapping)
    fn dx_transposed(&self) -> Self;
    /// Derivative on the Y axis (wrapping)
    fn dy(&self) -> Self;
    /// Derivative on the Y axis, transposed (wrapping)
    fn dy_transposed(&self) -> Self;
}

/// trait functions for exponentiation
pub trait Power {
    /// power of 2
    fn squared(&self) -> Self;
    /// power of n, where n is an unsigned 32-bit integer
    fn powi(&self, n: i32) -> Self;
    /// power of n, where n is a 64-bit floating point number
    fn powf(&self, n: f64) -> Self;
}
