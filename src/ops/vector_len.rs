use ndarray::{
    Array,
    Axis,
    Dimension,
    RemoveAxis,
    ShapeError,
};

/// Trait for calculating the lengths of two vectors
pub trait VectorLen: Sized {
    /// Calculates the vector lenght on the given axis for two inputs. The
    /// Output must be 1 dimension smaller.
    fn vector_len_on_axis(
        &self,
        other: &Self,
        axis: usize,
    ) -> Result<Self, ShapeError>;
}

impl<D: Dimension + RemoveAxis> VectorLen for Array<f64, D> {
    /// Calculates the vector lenght on the given axis for two inputs. The
    /// Output is 1 dimension smaller. In the context of images, for an axis Z
    /// holding the vector of colors, the output will be a grayscale image
    /// (the Z axis will be reduced to a single scalar value) of the same
    /// shape on the other axes.
    /// This is equivalent to `(Self^2 + Other^2).sum_axis(Z).map(|x|
    /// x.sqrt())`, where Self^2
    /// and Other^2 are the element-wise power of 2 on Self and Other,
    /// respectively, the .sum_axis(Z) reduces the array's axis Z into a
    /// scalar, and .map(|x| x.sqrt()) performs the  eleent-wise square
    /// root.
    fn vector_len_on_axis(
        &self,
        other: &Self,
        axis: usize,
    ) -> Result<Self, ShapeError> {
        if !(axis < self.ndim()) {
            let out_of_bounds = ndarray::ErrorKind::OutOfBounds;
            return Err(ShapeError::from_kind(out_of_bounds));
        }

        let mut vec_len = (self * self) + (other * other);
        if self.len_of(Axis(axis)) > 1 {
            vec_len.accumulate_axis_inplace(Axis(axis), |prev, curr| {
                *curr += prev
            });
            vec_len.collapse_axis(Axis(axis), vec_len.len_of(Axis(axis)) - 1);
        }
        vec_len.mapv_inplace(f64::sqrt);
        Ok(vec_len)
    }
}

#[cfg(test)]
mod test {
    use ndarray::{
        Array,
        Array3,
        Axis,
        ShapeError,
    };
    use pretty_assertions::assert_eq;

    use super::VectorLen;

    #[test]
    fn array_f64_vector_len_on_axis_returns_error_if_axis_is_out_of_bounds() {
        for dim in 0..7 {
            let shape: Vec<usize> = (0..dim).map(|x| x + 1).collect();
            let array = Array::<f64, _>::zeros(shape);

            let vec_len = array.vector_len_on_axis(&array, dim);

            let out_of_bounds_error =
                ShapeError::from_kind(ndarray::ErrorKind::OutOfBounds);
            assert_eq!(vec_len, Err(out_of_bounds_error));
        }
    }

    #[test]
    fn array_f64_vector_len_on_axis() {
        for z in 1..=4 {
            let mut a = Array3::zeros((10, 5, z));
            let mut b = Array3::zeros((10, 5, z));
            a.mapv_inplace(|_| rand::random::<u8>() as f64);
            b.mapv_inplace(|_| rand::random::<u8>() as f64);

            let len_of_vecs = a.vector_len_on_axis(&b, 2).unwrap();

            let test_len_of_vecs = ((&a * &a) + (&b * &b))
                .map_axis(Axis(2), |vector| vector.sum().sqrt());
            let test_len_of_vecs = test_len_of_vecs.insert_axis(Axis(2));

            assert_eq!(len_of_vecs, test_len_of_vecs);
        }
    }
}

#[cfg(test)]
mod bench {
    use ndarray::Array3;

    use super::VectorLen;

    #[bench]
    fn array_f64_vector_len_on_axis(bench: &mut test::Bencher) {
        let mut a = Array3::zeros((1024, 768, 3));
        let mut b = Array3::zeros((1024, 768, 3));
        a.mapv_inplace(|_| rand::random::<u8>() as f64);
        b.mapv_inplace(|_| rand::random::<u8>() as f64);

        bench.iter(|| test::black_box(a.vector_len_on_axis(&b, 2).unwrap()));
    }
}
