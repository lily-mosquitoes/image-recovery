use ndarray::{
    Array,
    Dimension,
};

/// Trait for calculating the Euclidean Norm of an array
pub trait Norm {
    fn norm(&self) -> f64;
}

impl<D: Dimension> Norm for Array<f64, D> {
    /// Calculates the Euclidean Norm of a vector,
    /// equivalent to `(self * self).sum().sqrt()`.
    fn norm(&self) -> f64 {
        (self * self).sum().sqrt()
    }
}

#[cfg(test)]
mod test {
    use ndarray::Array3;
    use pretty_assertions::assert_eq;

    use super::Norm;

    #[test]
    fn array_f64_norm() {
        let mut test_array = Array3::zeros((10, 5, 3));
        test_array.mapv_inplace(|_| rand::random::<f64>());

        let norm = test_array.norm();

        let test_norm = (&test_array * &test_array).sum().sqrt();

        assert_eq!(norm, test_norm);
    }
}
