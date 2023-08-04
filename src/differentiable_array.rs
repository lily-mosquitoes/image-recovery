use std::ops::Sub;

use ndarray::{
    Array,
    Axis,
    Dimension,
    RemoveAxis,
    ShapeError,
};

pub trait DifferentiableArray: Sized + Clone + Sub<Output = Self> {
    fn positive_shift_on_axis(&self, axis: usize) -> Result<Self, ShapeError>;

    fn positive_gradient_on_axis(
        &self,
        axis: usize,
    ) -> Result<Self, ShapeError> {
        let shifted = self.positive_shift_on_axis(axis)?;

        Ok(self.clone() - shifted)
    }

    fn negative_shift_on_axis(&self, axis: usize) -> Result<Self, ShapeError>;

    fn negative_gradient_on_axis(
        &self,
        axis: usize,
    ) -> Result<Self, ShapeError> {
        let shifted = self.negative_shift_on_axis(axis)?;

        Ok(self.clone() - shifted)
    }
}

impl<D: Dimension + RemoveAxis> DifferentiableArray for Array<f64, D> {
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

    use crate::differentiable_array::DifferentiableArray;

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
