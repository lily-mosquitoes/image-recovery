use ndarray::{
    Array,
    Dimension,
};

/// Trait for calculating the weighted average of two arrays, given some scalars
/// tau and lambda
pub trait Average {
    fn weighted_average(&self, other: &Self, tau: f64, lambda: f64) -> Self;
}

impl<D: Dimension> Average for Array<f64, D> {
    /// Calculates the weighted average of two arrays given some scalars tau and
    /// lambda, equivalent to `(other + (tau * lambda * self)) / (1.0 + tau
    /// * lambda).`
    fn weighted_average(&self, other: &Self, tau: f64, lambda: f64) -> Self {
        (other + (tau * lambda * self)) / (1.0 + tau * lambda)
    }
}

#[cfg(test)]
mod test {
    use ndarray::Array3;
    use pretty_assertions::assert_eq;

    use super::Average;

    #[test]
    fn array_f64_weighted_average() {
        let mut a = Array3::zeros((10, 5, 3));
        let mut b = Array3::zeros((10, 5, 3));
        a.mapv_inplace(|_| rand::random::<f64>());
        b.mapv_inplace(|_| rand::random::<f64>());

        let tau: f64 = 1.0 / 2_f64.sqrt();
        let lambda: f64 = 0.008;

        let average = a.weighted_average(&b, tau, lambda);

        let test_average = (&b + (tau * lambda * &a)) / (1.0 + tau * lambda);

        assert_eq!(average, test_average);
    }
}
