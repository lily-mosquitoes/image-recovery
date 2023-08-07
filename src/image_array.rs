use std::ops::Deref;

use image::{
    GrayImage,
    Luma,
    Rgb,
    RgbImage,
};
use ndarray::{
    Array,
    Array3,
    Axis,
    Dimension,
    RemoveAxis,
};

use crate::ops::{
    Average,
    Gradient,
    Norm,
    VectorLen,
};

/// An array representing an image, used with the solvers.
/// The From trait is implemented for the types GrayImage and RgbImage in the
/// [`image`](docs.rs/image/latest/image/) crate.
#[derive(Debug)]
pub struct ImageArray<T: Gradient + Average + VectorLen + Norm> {
    inner: T,
}

impl<T: Gradient + Average + VectorLen + Norm> Deref for ImageArray<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl From<&GrayImage> for ImageArray<Array3<f64>> {
    fn from(value: &GrayImage) -> Self {
        let dim = (value.width() as usize, value.height() as usize, 1);
        let mut array = Array3::<f64>::zeros(dim);
        for x in 0..dim.0 {
            for y in 0..dim.1 {
                let pixel = value.get_pixel(x as u32, y as u32);
                array[[x, y, 0]] = pixel[0] as f64;
            }
        }

        Self { inner: array }
    }
}

impl From<&RgbImage> for ImageArray<Array3<f64>> {
    fn from(value: &RgbImage) -> Self {
        let dim = (value.width() as usize, value.height() as usize, 3);
        let mut array = Array3::<f64>::zeros(dim);
        for x in 0..dim.0 {
            for y in 0..dim.1 {
                let pixel = value.get_pixel(x as u32, y as u32);
                for z in 0..3 {
                    array[[x, y, z]] = pixel[z] as f64;
                }
            }
        }

        Self { inner: array }
    }
}

impl<T: Copy + Into<f64>, D: Dimension + RemoveAxis> From<&Array<T, D>>
    for ImageArray<Array<f64, D>>
{
    fn from(value: &Array<T, D>) -> Self {
        Self {
            inner: value.map(|&v| <T as Into<f64>>::into(v)),
        }
    }
}

impl ImageArray<Array3<f64>> {
    /// Assumes Array3 axis 2 is colors, will flatten axis 2 if bigger than 1.
    pub fn into_luma(&self) -> GrayImage {
        let shape = self.shape();
        let flat = self.map_axis(Axis(2), |v| v.sum() as u8);
        let mut img = GrayImage::new(shape[0] as u32, shape[1] as u32);
        for x in 0..shape[0] {
            for y in 0..shape[1] {
                let pixel = Luma([flat[[x, y]]]);
                img.put_pixel(x as u32, y as u32, pixel);
            }
        }
        img
    }

    /// Assumes Array3 axis 2 is colors, will use 3 first elements of axis 2 if
    /// bigger than 3, or cycle through the elements if smaller than 3.
    pub fn into_rgb(&self) -> RgbImage {
        let shape = self.shape();
        let flat = self.map_axis(Axis(2), |v| v.map(|&x| x as u8).to_vec());
        let mut img = RgbImage::new(shape[0] as u32, shape[1] as u32);
        for x in 0..shape[0] {
            for y in 0..shape[1] {
                let mut colors = flat[[x, y]].iter().cloned().cycle();
                let pixel = Rgb([
                    colors.next().unwrap(),
                    colors.next().unwrap(),
                    colors.next().unwrap(),
                ]);
                img.put_pixel(x as u32, y as u32, pixel);
            }
        }
        img
    }
}

#[cfg(test)]
mod test {
    use image::{
        GrayImage,
        Luma,
        Rgb,
        RgbImage,
    };
    use ndarray::Array3;
    use pretty_assertions::assert_eq;

    use super::ImageArray;

    fn make_random_gray_image(shape: (u32, u32)) -> GrayImage {
        let mut img = GrayImage::new(shape.0, shape.1);
        for x in 0..shape.0 {
            for y in 0..shape.1 {
                let pixel = Luma(rand::random::<[u8; 1]>());
                img.put_pixel(x, y, pixel);
            }
        }
        img
    }

    fn make_random_rgb_image(shape: (u32, u32)) -> RgbImage {
        let mut img = RgbImage::new(shape.0, shape.1);
        for x in 0..shape.0 {
            for y in 0..shape.1 {
                let pixel = Rgb(rand::random::<[u8; 3]>());
                img.put_pixel(x, y, pixel);
            }
        }
        img
    }

    #[test]
    fn make_image_array_from_gray_image() {
        let img = make_random_gray_image((10, 5));

        let array = ImageArray::from(&img);

        let dim = (img.width() as usize, img.height() as usize, 1);
        let mut test_array = Array3::<f64>::zeros(dim);
        for x in 0..dim.0 {
            for y in 0..dim.1 {
                let pixel = img.get_pixel(x as u32, y as u32);
                test_array[[x, y, 0]] = pixel[0] as f64;
            }
        }

        assert_eq!(*array, test_array);
    }

    #[test]
    fn make_image_array_from_rgb_image() {
        let img = make_random_rgb_image((10, 5));

        let array = ImageArray::from(&img);

        let dim = (img.width() as usize, img.height() as usize, 3);
        let mut test_array = Array3::<f64>::zeros(dim);
        for x in 0..dim.0 {
            for y in 0..dim.1 {
                let pixel = img.get_pixel(x as u32, y as u32);
                for z in 0..3 {
                    test_array[[x, y, z]] = pixel[z] as f64;
                }
            }
        }

        assert_eq!(*array, test_array);
    }

    #[test]
    fn make_image_array_from_array3_u8() {
        let mut test_array = Array3::zeros((10, 5, 3));
        test_array.mapv_inplace(|_| rand::random::<u8>());

        let array = ImageArray::from(&test_array);

        let test_array = test_array.map(|&x| x as f64);

        assert_eq!(*array, test_array);
    }

    #[test]
    fn make_image_array_from_array3_f64() {
        let mut test_array = Array3::zeros((10, 5, 3));
        test_array.mapv_inplace(|_| rand::random::<f64>());

        let array = ImageArray::from(&test_array);

        assert_eq!(*array, test_array);
    }

    #[test]
    fn make_gray_image_from_array3_f64() {
        let test_img = make_random_gray_image((10, 5));

        let array = ImageArray::from(&test_img);
        let img = array.into_luma();

        assert_eq!(img, test_img);
    }

    #[test]
    fn make_rgb_image_from_array3_f64() {
        let test_img = make_random_rgb_image((10, 5));

        let array = ImageArray::from(&test_img);
        let img = array.into_rgb();

        assert_eq!(img, test_img);
    }
}
