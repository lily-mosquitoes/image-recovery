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

//! Implementation of Traits `Mul`, `Div`, `Add` and `Sub` for `RgbMatrices`.

use std::ops::{Mul, Div, Add, Sub};
use ndarray::Array2;
use crate::RgbMatrices;

// helper function
fn arr2_shape(x: &Array2<f64>) -> (usize, usize) {
    (x.ncols() as usize, x.nrows() as usize)
}

// impl Mul
impl Mul<RgbMatrices> for RgbMatrices {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        if self.shape != rhs.shape {
            panic!("icompatible shapes, self = {:?} x rhs = {:?}", self.shape, rhs.shape);
        }

        RgbMatrices {
            shape: self.shape,
            red: self.red * rhs.red,
            green: self.green * rhs.green,
            blue: self.blue * rhs.blue,
        }
    }
}

impl<'a> Mul<&'a RgbMatrices> for RgbMatrices {
    type Output = Self;

    fn mul(self, rhs: &'a RgbMatrices) -> Self {
        if self.shape != rhs.shape {
            panic!("icompatible shapes, self = {:?} x rhs = {:?}", self.shape, rhs.shape);
        }

        RgbMatrices {
            shape: self.shape,
            red: self.red * rhs.red.to_owned(),
            green: self.green * rhs.green.to_owned(),
            blue: self.blue * rhs.blue.to_owned(),
        }
    }
}

impl<'a> Mul<RgbMatrices> for &'a RgbMatrices {
    type Output = RgbMatrices;

    fn mul(self, rhs: RgbMatrices) -> RgbMatrices {
        if self.shape != rhs.shape {
            panic!("icompatible shapes, self = {:?} x rhs = {:?}", self.shape, rhs.shape);
        }

        RgbMatrices {
            shape: self.shape,
            red: self.red.to_owned() * rhs.red,
            green: self.green.to_owned() * rhs.green,
            blue: self.blue.to_owned() * rhs.blue,
        }
    }
}

impl<'a> Mul<&'a RgbMatrices> for &'a RgbMatrices {
    type Output = RgbMatrices;

    fn mul(self, rhs: &'a RgbMatrices) -> RgbMatrices {
        if self.shape != rhs.shape {
            panic!("icompatible shapes, self = {:?} x rhs = {:?}", self.shape, rhs.shape);
        }

        RgbMatrices {
            shape: self.shape,
            red: self.red.to_owned() * rhs.red.to_owned(),
            green: self.green.to_owned() * rhs.green.to_owned(),
            blue: self.blue.to_owned() * rhs.blue.to_owned(),
        }
    }
}

impl Mul<Array2<f64>> for RgbMatrices {
    type Output = Self;

    fn mul(self, rhs: Array2<f64>) -> Self {
        if self.shape != arr2_shape(&rhs) {
            panic!("icompatible shapes, self = {:?} x rhs = {:?}", self.shape, rhs.shape());
        }

        RgbMatrices {
            shape: self.shape,
            red: self.red * rhs.to_owned(),
            green: self.green * rhs.to_owned(),
            blue: self.blue * rhs.to_owned(),
        }
    }
}

impl<'a> Mul<&'a Array2<f64>> for RgbMatrices {
    type Output = Self;

    fn mul(self, rhs: &'a Array2<f64>) -> Self {
        if self.shape != arr2_shape(rhs) {
            panic!("icompatible shapes, self = {:?} x rhs = {:?}", self.shape, rhs.shape());
        }

        RgbMatrices {
            shape: self.shape,
            red: self.red * rhs,
            green: self.green * rhs,
            blue: self.blue * rhs,
        }
    }
}

impl<'a> Mul<Array2<f64>> for &'a RgbMatrices {
    type Output = RgbMatrices;

    fn mul(self, rhs: Array2<f64>) -> RgbMatrices {
        if self.shape != arr2_shape(&rhs) {
            panic!("icompatible shapes, self = {:?} x rhs = {:?}", self.shape, rhs.shape());
        }

        RgbMatrices {
            shape: self.shape,
            red: self.red.to_owned() * rhs.to_owned(),
            green: self.green.to_owned() * rhs.to_owned(),
            blue: self.blue.to_owned() * rhs.to_owned(),
        }
    }
}

impl<'a> Mul<&'a Array2<f64>> for &'a RgbMatrices {
    type Output = RgbMatrices;

    fn mul(self, rhs: &'a Array2<f64>) -> RgbMatrices {
        if self.shape != arr2_shape(rhs) {
            panic!("icompatible shapes, self = {:?} x rhs = {:?}", self.shape, rhs.shape());
        }

        RgbMatrices {
            shape: self.shape,
            red: self.red.to_owned() * rhs,
            green: self.green.to_owned() * rhs,
            blue: self.blue.to_owned() * rhs,
        }
    }
}

impl Mul<f64> for RgbMatrices {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self {
        RgbMatrices {
            shape: self.shape,
            red: self.red * rhs,
            green: self.green * rhs,
            blue: self.blue * rhs,
        }
    }
}

impl<'a> Mul<f64> for &'a RgbMatrices {
    type Output = RgbMatrices;

    fn mul(self, rhs: f64) -> RgbMatrices {
        RgbMatrices {
            shape: self.shape,
            red: self.red.to_owned() * rhs,
            green: self.green.to_owned() * rhs,
            blue: self.blue.to_owned() * rhs,
        }
    }
}

impl Mul<RgbMatrices> for f64 {
    type Output = RgbMatrices;

    fn mul(self, rhs: RgbMatrices) -> RgbMatrices {
        RgbMatrices {
            shape: rhs.shape,
            red: rhs.red * self,
            green: rhs.green * self,
            blue: rhs.blue * self,
        }
    }
}

impl<'a> Mul<&'a RgbMatrices> for f64 {
    type Output = RgbMatrices;

    fn mul(self, rhs: &'a RgbMatrices) -> RgbMatrices {
        RgbMatrices {
            shape: rhs.shape,
            red: rhs.red.to_owned() * self,
            green: rhs.green.to_owned() * self,
            blue: rhs.blue.to_owned() * self,
        }
    }
}

// impl Div
impl Div<RgbMatrices> for RgbMatrices {
    type Output = Self;

    fn div(self, rhs: Self) -> Self {
        if self.shape != rhs.shape {
            panic!("icompatible shapes, self = {:?} x rhs = {:?}", self.shape, rhs.shape);
        }

        RgbMatrices {
            shape: self.shape,
            red: self.red / rhs.red,
            green: self.green / rhs.green,
            blue: self.blue / rhs.blue,
        }
    }
}

impl<'a> Div<&'a RgbMatrices> for RgbMatrices {
    type Output = Self;

    fn div(self, rhs: &'a RgbMatrices) -> Self {
        if self.shape != rhs.shape {
            panic!("icompatible shapes, self = {:?} x rhs = {:?}", self.shape, rhs.shape);
        }

        RgbMatrices {
            shape: self.shape,
            red: self.red / rhs.red.to_owned(),
            green: self.green / rhs.green.to_owned(),
            blue: self.blue / rhs.blue.to_owned(),
        }
    }
}

impl<'a> Div<RgbMatrices> for &'a RgbMatrices {
    type Output = RgbMatrices;

    fn div(self, rhs: RgbMatrices) -> RgbMatrices {
        if self.shape != rhs.shape {
            panic!("icompatible shapes, self = {:?} x rhs = {:?}", self.shape, rhs.shape);
        }

        RgbMatrices {
            shape: self.shape,
            red: self.red.to_owned() / rhs.red,
            green: self.green.to_owned() / rhs.green,
            blue: self.blue.to_owned() / rhs.blue,
        }
    }
}

impl<'a> Div<&'a RgbMatrices> for &'a RgbMatrices {
    type Output = RgbMatrices;

    fn div(self, rhs: &'a RgbMatrices) -> RgbMatrices {
        if self.shape != rhs.shape {
            panic!("icompatible shapes, self = {:?} x rhs = {:?}", self.shape, rhs.shape);
        }

        RgbMatrices {
            shape: self.shape,
            red: self.red.to_owned() / rhs.red.to_owned(),
            green: self.green.to_owned() / rhs.green.to_owned(),
            blue: self.blue.to_owned() / rhs.blue.to_owned(),
        }
    }
}

impl Div<Array2<f64>> for RgbMatrices {
    type Output = Self;

    fn div(self, rhs: Array2<f64>) -> Self {
        if self.shape != arr2_shape(&rhs) {
            panic!("icompatible shapes, self = {:?} x rhs = {:?}", self.shape, rhs.shape());
        }

        RgbMatrices {
            shape: self.shape,
            red: self.red / rhs.to_owned(),
            green: self.green / rhs.to_owned(),
            blue: self.blue / rhs.to_owned(),
        }
    }
}

impl<'a> Div<&'a Array2<f64>> for RgbMatrices {
    type Output = Self;

    fn div(self, rhs: &'a Array2<f64>) -> Self {
        if self.shape != arr2_shape(rhs) {
            panic!("icompatible shapes, self = {:?} x rhs = {:?}", self.shape, rhs.shape());
        }

        RgbMatrices {
            shape: self.shape,
            red: self.red / rhs,
            green: self.green / rhs,
            blue: self.blue / rhs,
        }
    }
}

impl<'a> Div<Array2<f64>> for &'a RgbMatrices {
    type Output = RgbMatrices;

    fn div(self, rhs: Array2<f64>) -> RgbMatrices {
        if self.shape != arr2_shape(&rhs) {
            panic!("icompatible shapes, self = {:?} x rhs = {:?}", self.shape, rhs.shape());
        }

        RgbMatrices {
            shape: self.shape,
            red: self.red.to_owned() / rhs.to_owned(),
            green: self.green.to_owned() / rhs.to_owned(),
            blue: self.blue.to_owned() / rhs.to_owned(),
        }
    }
}

impl<'a> Div<&'a Array2<f64>> for &'a RgbMatrices {
    type Output = RgbMatrices;

    fn div(self, rhs: &'a Array2<f64>) -> RgbMatrices {
        if self.shape != arr2_shape(rhs) {
            panic!("icompatible shapes, self = {:?} x rhs = {:?}", self.shape, rhs.shape());
        }

        RgbMatrices {
            shape: self.shape,
            red: self.red.to_owned() / rhs,
            green: self.green.to_owned() / rhs,
            blue: self.blue.to_owned() / rhs,
        }
    }
}

impl Div<f64> for RgbMatrices {
    type Output = Self;

    fn div(self, rhs: f64) -> Self {
        RgbMatrices {
            shape: self.shape,
            red: self.red / rhs,
            green: self.green / rhs,
            blue: self.blue / rhs,
        }
    }
}

impl<'a> Div<f64> for &'a RgbMatrices {
    type Output = RgbMatrices;

    fn div(self, rhs: f64) -> RgbMatrices {
        RgbMatrices {
            shape: self.shape,
            red: self.red.to_owned() / rhs,
            green: self.green.to_owned() / rhs,
            blue: self.blue.to_owned() / rhs,
        }
    }
}

impl Div<RgbMatrices> for f64 {
    type Output = RgbMatrices;

    fn div(self, rhs: RgbMatrices) -> RgbMatrices {
        RgbMatrices {
            shape: rhs.shape,
            red: rhs.red / self,
            green: rhs.green / self,
            blue: rhs.blue / self,
        }
    }
}

impl<'a> Div<&'a RgbMatrices> for f64 {
    type Output = RgbMatrices;

    fn div(self, rhs: &'a RgbMatrices) -> RgbMatrices {
        RgbMatrices {
            shape: rhs.shape,
            red: rhs.red.to_owned() / self,
            green: rhs.green.to_owned() / self,
            blue: rhs.blue.to_owned() / self,
        }
    }
}

// impl Add
impl Add<RgbMatrices> for RgbMatrices {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        if self.shape != rhs.shape {
            panic!("icompatible shapes, self = {:?} x rhs = {:?}", self.shape, rhs.shape);
        }

        RgbMatrices {
            shape: self.shape,
            red: self.red + rhs.red,
            green: self.green + rhs.green,
            blue: self.blue + rhs.blue,
        }
    }
}

impl<'a> Add<&'a RgbMatrices> for RgbMatrices {
    type Output = Self;

    fn add(self, rhs: &'a RgbMatrices) -> Self {
        if self.shape != rhs.shape {
            panic!("icompatible shapes, self = {:?} x rhs = {:?}", self.shape, rhs.shape);
        }

        RgbMatrices {
            shape: self.shape,
            red: self.red + rhs.red.to_owned(),
            green: self.green + rhs.green.to_owned(),
            blue: self.blue + rhs.blue.to_owned(),
        }
    }
}

impl<'a> Add<RgbMatrices> for &'a RgbMatrices {
    type Output = RgbMatrices;

    fn add(self, rhs: RgbMatrices) -> RgbMatrices {
        if self.shape != rhs.shape {
            panic!("icompatible shapes, self = {:?} x rhs = {:?}", self.shape, rhs.shape);
        }

        RgbMatrices {
            shape: self.shape,
            red: self.red.to_owned() + rhs.red,
            green: self.green.to_owned() + rhs.green,
            blue: self.blue.to_owned() + rhs.blue,
        }
    }
}

impl<'a> Add<&'a RgbMatrices> for &'a RgbMatrices {
    type Output = RgbMatrices;

    fn add(self, rhs: &'a RgbMatrices) -> RgbMatrices {
        if self.shape != rhs.shape {
            panic!("icompatible shapes, self = {:?} x rhs = {:?}", self.shape, rhs.shape);
        }

        RgbMatrices {
            shape: self.shape,
            red: self.red.to_owned() + rhs.red.to_owned(),
            green: self.green.to_owned() + rhs.green.to_owned(),
            blue: self.blue.to_owned() + rhs.blue.to_owned(),
        }
    }
}

impl Add<Array2<f64>> for RgbMatrices {
    type Output = Self;

    fn add(self, rhs: Array2<f64>) -> Self {
        if self.shape != arr2_shape(&rhs) {
            panic!("icompatible shapes, self = {:?} x rhs = {:?}", self.shape, rhs.shape());
        }

        RgbMatrices {
            shape: self.shape,
            red: self.red + rhs.to_owned(),
            green: self.green + rhs.to_owned(),
            blue: self.blue + rhs.to_owned(),
        }
    }
}

impl<'a> Add<&'a Array2<f64>> for RgbMatrices {
    type Output = Self;

    fn add(self, rhs: &'a Array2<f64>) -> Self {
        if self.shape != arr2_shape(rhs) {
            panic!("icompatible shapes, self = {:?} x rhs = {:?}", self.shape, rhs.shape());
        }

        RgbMatrices {
            shape: self.shape,
            red: self.red + rhs,
            green: self.green + rhs,
            blue: self.blue + rhs,
        }
    }
}

impl<'a> Add<Array2<f64>> for &'a RgbMatrices {
    type Output = RgbMatrices;

    fn add(self, rhs: Array2<f64>) -> RgbMatrices {
        if self.shape != arr2_shape(&rhs) {
            panic!("icompatible shapes, self = {:?} x rhs = {:?}", self.shape, rhs.shape());
        }

        RgbMatrices {
            shape: self.shape,
            red: self.red.to_owned() + rhs.to_owned(),
            green: self.green.to_owned() + rhs.to_owned(),
            blue: self.blue.to_owned() + rhs.to_owned(),
        }
    }
}

impl<'a> Add<&'a Array2<f64>> for &'a RgbMatrices {
    type Output = RgbMatrices;

    fn add(self, rhs: &'a Array2<f64>) -> RgbMatrices {
        if self.shape != arr2_shape(rhs) {
            panic!("icompatible shapes, self = {:?} x rhs = {:?}", self.shape, rhs.shape());
        }

        RgbMatrices {
            shape: self.shape,
            red: self.red.to_owned() + rhs,
            green: self.green.to_owned() + rhs,
            blue: self.blue.to_owned() + rhs,
        }
    }
}

impl Add<f64> for RgbMatrices {
    type Output = Self;

    fn add(self, rhs: f64) -> Self {
        RgbMatrices {
            shape: self.shape,
            red: self.red + rhs,
            green: self.green + rhs,
            blue: self.blue + rhs,
        }
    }
}

impl<'a> Add<f64> for &'a RgbMatrices {
    type Output = RgbMatrices;

    fn add(self, rhs: f64) -> RgbMatrices {
        RgbMatrices {
            shape: self.shape,
            red: self.red.to_owned() + rhs,
            green: self.green.to_owned() + rhs,
            blue: self.blue.to_owned() + rhs,
        }
    }
}

impl Add<RgbMatrices> for f64 {
    type Output = RgbMatrices;

    fn add(self, rhs: RgbMatrices) -> RgbMatrices {
        RgbMatrices {
            shape: rhs.shape,
            red: rhs.red + self,
            green: rhs.green + self,
            blue: rhs.blue + self,
        }
    }
}

impl<'a> Add<&'a RgbMatrices> for f64 {
    type Output = RgbMatrices;

    fn add(self, rhs: &'a RgbMatrices) -> RgbMatrices {
        RgbMatrices {
            shape: rhs.shape,
            red: rhs.red.to_owned() + self,
            green: rhs.green.to_owned() + self,
            blue: rhs.blue.to_owned() + self,
        }
    }
}

// impl Sub
impl Sub<RgbMatrices> for RgbMatrices {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        if self.shape != rhs.shape {
            panic!("icompatible shapes, self = {:?} x rhs = {:?}", self.shape, rhs.shape);
        }

        RgbMatrices {
            shape: self.shape,
            red: self.red - rhs.red,
            green: self.green - rhs.green,
            blue: self.blue - rhs.blue,
        }
    }
}

impl<'a> Sub<&'a RgbMatrices> for RgbMatrices {
    type Output = Self;

    fn sub(self, rhs: &'a RgbMatrices) -> Self {
        if self.shape != rhs.shape {
            panic!("icompatible shapes, self = {:?} x rhs = {:?}", self.shape, rhs.shape);
        }

        RgbMatrices {
            shape: self.shape,
            red: self.red - rhs.red.to_owned(),
            green: self.green - rhs.green.to_owned(),
            blue: self.blue - rhs.blue.to_owned(),
        }
    }
}

impl<'a> Sub<RgbMatrices> for &'a RgbMatrices {
    type Output = RgbMatrices;

    fn sub(self, rhs: RgbMatrices) -> RgbMatrices {
        if self.shape != rhs.shape {
            panic!("icompatible shapes, self = {:?} x rhs = {:?}", self.shape, rhs.shape);
        }

        RgbMatrices {
            shape: self.shape,
            red: self.red.to_owned() - rhs.red,
            green: self.green.to_owned() - rhs.green,
            blue: self.blue.to_owned() - rhs.blue,
        }
    }
}

impl<'a> Sub<&'a RgbMatrices> for &'a RgbMatrices {
    type Output = RgbMatrices;

    fn sub(self, rhs: &'a RgbMatrices) -> RgbMatrices {
        if self.shape != rhs.shape {
            panic!("icompatible shapes, self = {:?} x rhs = {:?}", self.shape, rhs.shape);
        }

        RgbMatrices {
            shape: self.shape,
            red: self.red.to_owned() - rhs.red.to_owned(),
            green: self.green.to_owned() - rhs.green.to_owned(),
            blue: self.blue.to_owned() - rhs.blue.to_owned(),
        }
    }
}

impl Sub<Array2<f64>> for RgbMatrices {
    type Output = Self;

    fn sub(self, rhs: Array2<f64>) -> Self {
        if self.shape != arr2_shape(&rhs) {
            panic!("icompatible shapes, self = {:?} x rhs = {:?}", self.shape, rhs.shape());
        }

        RgbMatrices {
            shape: self.shape,
            red: self.red - rhs.to_owned(),
            green: self.green - rhs.to_owned(),
            blue: self.blue - rhs.to_owned(),
        }
    }
}

impl<'a> Sub<&'a Array2<f64>> for RgbMatrices {
    type Output = Self;

    fn sub(self, rhs: &'a Array2<f64>) -> Self {
        if self.shape != arr2_shape(rhs) {
            panic!("icompatible shapes, self = {:?} x rhs = {:?}", self.shape, rhs.shape());
        }

        RgbMatrices {
            shape: self.shape,
            red: self.red - rhs,
            green: self.green - rhs,
            blue: self.blue - rhs,
        }
    }
}

impl<'a> Sub<Array2<f64>> for &'a RgbMatrices {
    type Output = RgbMatrices;

    fn sub(self, rhs: Array2<f64>) -> RgbMatrices {
        if self.shape != arr2_shape(&rhs) {
            panic!("icompatible shapes, self = {:?} x rhs = {:?}", self.shape, rhs.shape());
        }

        RgbMatrices {
            shape: self.shape,
            red: self.red.to_owned() - rhs.to_owned(),
            green: self.green.to_owned() - rhs.to_owned(),
            blue: self.blue.to_owned() - rhs.to_owned(),
        }
    }
}

impl<'a> Sub<&'a Array2<f64>> for &'a RgbMatrices {
    type Output = RgbMatrices;

    fn sub(self, rhs: &'a Array2<f64>) -> RgbMatrices {
        if self.shape != arr2_shape(rhs) {
            panic!("icompatible shapes, self = {:?} x rhs = {:?}", self.shape, rhs.shape());
        }

        RgbMatrices {
            shape: self.shape,
            red: self.red.to_owned() - rhs,
            green: self.green.to_owned() - rhs,
            blue: self.blue.to_owned() - rhs,
        }
    }
}

impl Sub<f64> for RgbMatrices {
    type Output = Self;

    fn sub(self, rhs: f64) -> Self {
        RgbMatrices {
            shape: self.shape,
            red: self.red - rhs,
            green: self.green - rhs,
            blue: self.blue - rhs,
        }
    }
}

impl<'a> Sub<f64> for &'a RgbMatrices {
    type Output = RgbMatrices;

    fn sub(self, rhs: f64) -> RgbMatrices {
        RgbMatrices {
            shape: self.shape,
            red: self.red.to_owned() - rhs,
            green: self.green.to_owned() - rhs,
            blue: self.blue.to_owned() - rhs,
        }
    }
}

impl Sub<RgbMatrices> for f64 {
    type Output = RgbMatrices;

    fn sub(self, rhs: RgbMatrices) -> RgbMatrices {
        RgbMatrices {
            shape: rhs.shape,
            red: rhs.red - self,
            green: rhs.green - self,
            blue: rhs.blue - self,
        }
    }
}

impl<'a> Sub<&'a RgbMatrices> for f64 {
    type Output = RgbMatrices;

    fn sub(self, rhs: &'a RgbMatrices) -> RgbMatrices {
        RgbMatrices {
            shape: rhs.shape,
            red: rhs.red.to_owned() - self,
            green: rhs.green.to_owned() - self,
            blue: rhs.blue.to_owned() - self,
        }
    }
}
