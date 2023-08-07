use std::ops::Sub;

use ndarray::{
    Array,
    Axis,
    Dimension,
    RemoveAxis,
    ShapeError,
};

/// Trait for calculating the gradient (derivation) on an axis of a N
/// dimentional Array. The gradient methods are provided using the shift methods
/// for Self which implements &Self - &Self. The gradient must be implemented
/// such that for all X, (PG_A * B).sum() == A * NG_B.sum(), where A and B
/// are arrays of the same shape, PG_A is the positive gradient of A on some
/// axis X and NG_B is the negative gradient of B on that same axis X.
pub trait Gradient: Sized {
    /// Must output a same shape array shifted towards the growing indexes on
    /// the given axis. On the boundary, the shift must be wrapping (i.e. the
    /// last index of the given axis will become the 0th index). Must be checked
    /// for bounds (i.e. given axis must exist in array) and size of the
    /// given axis, as a shift cannot be performed on an axis with len < 2.
    fn positive_shift_on_axis(&self, axis: usize) -> Result<Self, ShapeError>;

    /// Outputs the same shape array by shifting on the given axis and
    /// subtracting the result from self. Returns any error from shifting,
    /// which must be checked for bounds (i.e. given axis must exist in
    /// array) adnd size of the given axis (must be > 2). The gradient is
    /// implemented such that for all X, (PG_A * B).sum() == A * NG_B.sum(),
    /// where A and B are arrays of the same shape, PG_A is the positive
    /// gradient of A on some axis X, NG_B is the negative gradient of B
    /// on that same axis X, and .sum() returns a scalar with the sum of all
    /// elements of the array.
    fn positive_gradient_on_axis(&self, axis: usize) -> Result<Self, ShapeError>
    where
        for<'x> &'x Self: Sub<Output = Self>,
    {
        let shifted = self.positive_shift_on_axis(axis)?;

        Ok(self - &shifted)
    }

    /// Must output a same shape array shifted towards the shrinking indexes on
    /// the given axis. On the boundary, the shift must be wrapping (i.e. the
    /// 0th index of the given axis will become the last index). Must be checked
    /// for bounds (i.e. given axis must exist in array) and size of the
    /// given axis, as a shift cannot be performed on an axis with len < 2.
    fn negative_shift_on_axis(&self, axis: usize) -> Result<Self, ShapeError>;

    /// Outputs the same shape array by shifting on the given axis and
    /// subtracting the result from self. Returns any error from shifting,
    /// which must be checked for bounds (i.e. given axis must exist in
    /// array) adnd size of the given axis (must be > 2). The gradient is
    /// implemented such that for all X, (PG_A * B).sum() == A * NG_B.sum(),
    /// where A and B are arrays of the same shape, PG_A is the positive
    /// gradient of A on some axis X, NG_B is the negative gradient of B
    /// on that same axis X, and .sum() returns a scalar with the sum of all
    /// elements of the array.
    fn negative_gradient_on_axis(&self, axis: usize) -> Result<Self, ShapeError>
    where
        for<'x> &'x Self: Sub<Output = Self>,
    {
        let shifted = self.negative_shift_on_axis(axis)?;

        Ok(self - &shifted)
    }
}

impl<D: Dimension + RemoveAxis> Gradient for Array<f64, D> {
    /// Outputs a same shape array shifted towards the growing indexes on
    /// the given axis. On the boundary, the shift is wrapping (i.e. the
    /// last index of the given axis will become the 0th index). The input is
    /// checked for bounds (i.e. given axis must exist in array) and size of
    /// the given axis, as a shift cannot be performed on an axis with len <
    /// 2.
    fn positive_shift_on_axis(&self, axis: usize) -> Result<Self, ShapeError> {
        if !(axis < self.ndim()) {
            let out_of_bounds = ndarray::ErrorKind::OutOfBounds;
            return Err(ShapeError::from_kind(out_of_bounds));
        }

        if !(self.len_of(Axis(axis)) > 1) {
            let unsupported = ndarray::ErrorKind::Unsupported;
            return Err(ShapeError::from_kind(unsupported));
        }

        let last_index_of_axis = self.len_of(Axis(axis)) - 1;
        let (a, b) = self.view().split_at(Axis(axis), last_index_of_axis);
        ndarray::concatenate(Axis(axis), &[b, a])
    }

    /// Outputs a same shape array shifted towards the shrinking indexes on
    /// the given axis. On the boundary, the shift is wrapping (i.e. the
    /// 0th index of the given axis will become the last index). The input is
    /// checked for bounds (i.e. given axis must exist in array) and size of
    /// the given axis, as a shift cannot be performed on an axis with len <
    /// 2.
    fn negative_shift_on_axis(&self, axis: usize) -> Result<Self, ShapeError> {
        if !(axis < self.ndim()) {
            let out_of_bounds = ndarray::ErrorKind::OutOfBounds;
            return Err(ShapeError::from_kind(out_of_bounds));
        }

        if !(self.len_of(Axis(axis)) > 1) {
            let unsupported = ndarray::ErrorKind::Unsupported;
            return Err(ShapeError::from_kind(unsupported));
        }

        let (a, b) = self.view().split_at(Axis(axis), 1);
        ndarray::concatenate(Axis(axis), &[b, a])
    }
}

#[cfg(test)]
mod test {
    use ndarray::{
        Array,
        Axis,
        ShapeError,
    };
    use pretty_assertions::assert_eq;
    use rand::seq::IteratorRandom;

    use super::Gradient;

    #[test]
    fn array_f64_positive_shift_on_axis_returns_error_if_axis_is_out_of_bounds()
    {
        for dim in 0..=7 {
            let shape: Vec<usize> = (0..dim).map(|x| x + 1).collect();
            let array = Array::<f64, _>::zeros(shape);

            let shifted = array.positive_shift_on_axis(dim);

            let out_of_bounds_error =
                ShapeError::from_kind(ndarray::ErrorKind::OutOfBounds);
            assert_eq!(shifted, Err(out_of_bounds_error));
        }
    }

    #[test]
    fn array_f64_positive_shift_on_axis_returns_error_if_axis_len_is_not_gt_1()
    {
        // Array0 has no axes
        for dim in 1..=7 {
            let shape: Vec<usize> = (0..dim).map(|x| x + 1).collect();
            let array = Array::<f64, _>::zeros(shape);

            let shifted = array.positive_shift_on_axis(0);

            let unsupported_error =
                ShapeError::from_kind(ndarray::ErrorKind::Unsupported);
            assert_eq!(shifted, Err(unsupported_error));
        }
    }

    #[test]
    fn array_f64_positive_shift_on_axis() {
        let mut rng = rand::thread_rng();
        // Shift only supported for axis len > 1
        let mut random_axis_len = || (2..10).choose(&mut rng).unwrap();

        // Array0 has no axes
        for dim in 1..=7 {
            let shape: Vec<usize> =
                (0..dim).map(|_| random_axis_len()).collect();
            let mut array = Array::<f64, _>::zeros(shape);
            array.mapv_inplace(|_| rand::random::<u8>() as f64);

            for axis in 0..dim {
                let shifted = array.positive_shift_on_axis(axis).unwrap();

                let last_index_of_x = array.len_of(Axis(axis)) - 1;
                let (a, b) = array.view().split_at(Axis(axis), last_index_of_x);
                let test_shifted =
                    ndarray::concatenate(Axis(axis), &[b, a]).unwrap();

                assert_eq!(shifted, test_shifted);
            }
        }
    }

    #[test]
    fn array_f64_positive_gradient_on_axis() {
        let mut rng = rand::thread_rng();
        // Shift only supported for axis len > 1
        let mut random_axis_len = || (2..10).choose(&mut rng).unwrap();

        // Array0 has no axes
        for dim in 1..=7 {
            let shape: Vec<usize> =
                (0..dim).map(|_| random_axis_len()).collect();
            let mut array = Array::<f64, _>::zeros(shape);
            array.mapv_inplace(|_| rand::random::<u8>() as f64);

            for axis in 0..dim {
                let gradient = array.positive_gradient_on_axis(axis).unwrap();

                let last_index_of_x = array.len_of(Axis(axis)) - 1;
                let (a, b) = array.view().split_at(Axis(axis), last_index_of_x);
                let test_shifted =
                    ndarray::concatenate(Axis(axis), &[b, a]).unwrap();
                let test_gradient = &array - test_shifted;

                assert_eq!(gradient, test_gradient);
            }
        }
    }

    #[test]
    fn array_f64_negative_shift_on_axis_returns_error_if_axis_is_out_of_bounds()
    {
        for dim in 0..=7 {
            let shape: Vec<usize> = (0..dim).map(|x| x + 1).collect();
            let array = Array::<f64, _>::zeros(shape);

            let shifted = array.negative_shift_on_axis(dim);

            let out_of_bounds_error =
                ShapeError::from_kind(ndarray::ErrorKind::OutOfBounds);
            assert_eq!(shifted, Err(out_of_bounds_error));
        }
    }

    #[test]
    fn array_f64_negative_shift_on_axis_returns_error_if_axis_len_is_not_gt_1()
    {
        // Array0 has no axes
        for dim in 1..=7 {
            let shape: Vec<usize> = (0..dim).map(|x| x + 1).collect();
            let array = Array::<f64, _>::zeros(shape);

            let shifted = array.negative_shift_on_axis(0);

            let unsupported_error =
                ShapeError::from_kind(ndarray::ErrorKind::Unsupported);
            assert_eq!(shifted, Err(unsupported_error));
        }
    }

    #[test]
    fn array_f64_negative_shift_on_axis() {
        let mut rng = rand::thread_rng();
        // Shift only supported for axis len > 1
        let mut random_axis_len = || (2..10).choose(&mut rng).unwrap();

        // Array0 has no axes
        for dim in 1..=7 {
            let shape: Vec<usize> =
                (0..dim).map(|_| random_axis_len()).collect();
            let mut array = Array::<f64, _>::zeros(shape);
            array.mapv_inplace(|_| rand::random::<u8>() as f64);

            for axis in 0..dim {
                let shifted = array.negative_shift_on_axis(axis).unwrap();

                let (a, b) = array.view().split_at(Axis(axis), 1);
                let test_shifted =
                    ndarray::concatenate(Axis(axis), &[b, a]).unwrap();

                assert_eq!(shifted, test_shifted);
            }
        }
    }

    #[test]
    fn array_f64_negative_gradient_on_axis() {
        let mut rng = rand::thread_rng();
        // Shift only supported for axis len > 1
        let mut random_axis_len = || (2..10).choose(&mut rng).unwrap();

        // Array0 has no axes
        for dim in 1..=7 {
            let shape: Vec<usize> =
                (0..dim).map(|_| random_axis_len()).collect();
            let mut array = Array::<f64, _>::zeros(shape);
            array.mapv_inplace(|_| rand::random::<u8>() as f64);

            for axis in 0..dim {
                let gradient = array.negative_gradient_on_axis(axis).unwrap();

                let (a, b) = array.view().split_at(Axis(axis), 1);
                let test_shifted =
                    ndarray::concatenate(Axis(axis), &[b, a]).unwrap();
                let test_gradient = &array - test_shifted;

                assert_eq!(gradient, test_gradient);
            }
        }
    }

    #[test]
    fn array_f64_negative_gradient_on_axis_is_dual_operator_of_positive_gradient_on_axis(
    ) {
        let mut rng = rand::thread_rng();
        // Shift only supported for axis len > 1
        let mut random_axis_len = || (2..10).choose(&mut rng).unwrap();

        // Array0 has no axes
        for dim in 1..=7 {
            let shape: Vec<usize> =
                (0..dim).map(|_| random_axis_len()).collect();
            let mut array_a = Array::<f64, _>::zeros(shape.clone());
            array_a.mapv_inplace(|_| rand::random::<u8>() as f64);
            let mut array_b = Array::<f64, _>::zeros(shape);
            array_b.mapv_inplace(|_| rand::random::<u8>() as f64);

            for axis in 0..dim {
                let pos_a = array_a.positive_gradient_on_axis(axis).unwrap();
                let neg_b = array_b.negative_gradient_on_axis(axis).unwrap();

                assert_eq!((pos_a * &array_b).sum(), (&array_a * neg_b).sum());
            }
        }
    }
}

#[cfg(test)]
mod bench {
    use ndarray::Array3;

    use super::Gradient;

    #[bench]
    fn array_f64_positive_gradient_on_axis(bench: &mut test::Bencher) {
        let mut a = Array3::zeros((1024, 768, 3));
        a.mapv_inplace(|_| rand::random::<u8>() as f64);

        bench.iter(|| test::black_box(a.positive_gradient_on_axis(2).unwrap()));
    }

    #[bench]
    fn array_f64_negative_gradient_on_axis(bench: &mut test::Bencher) {
        let mut a = Array3::zeros((1024, 768, 3));
        a.mapv_inplace(|_| rand::random::<u8>() as f64);

        bench.iter(|| test::black_box(a.negative_gradient_on_axis(2).unwrap()));
    }
}
