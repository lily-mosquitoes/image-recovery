use std::ops::Deref;

use image::{
    GrayImage,
    RgbImage,
};
use ndarray::{
    Array,
    Array2,
    Array3,
    Dimension,
    RemoveAxis,
};

use crate::differentiable_array::DifferentiableArray;

#[derive(Debug)]
pub struct ImageArray<T: DifferentiableArray> {
    inner: T,
}

impl<T: DifferentiableArray> Deref for ImageArray<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl From<&GrayImage> for ImageArray<Array2<f64>> {
    fn from(value: &GrayImage) -> Self {
        let dim = (value.width() as usize, value.height() as usize);
        let mut array = Array2::<f64>::zeros(dim);
        for x in 0..dim.0 {
            for y in 0..dim.1 {
                let pixel = value.get_pixel(x as u32, y as u32);
                array[[x, y]] = pixel[0] as f64;
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

#[cfg(test)]
mod test {
    use image::{
        GrayImage,
        Luma,
        Rgb,
        RgbImage,
    };
    use ndarray::{
        Array2,
        Array3,
    };
    use pretty_assertions::assert_eq;

    use super::ImageArray;

    fn get_gray_image(shape: (u32, u32)) -> GrayImage {
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
        let img = get_gray_image((10, 5));

        let array = ImageArray::from(&img);

        let dim = (img.width() as usize, img.height() as usize);
        let mut test_array = Array2::<f64>::zeros(dim);
        for x in 0..dim.0 {
            for y in 0..dim.1 {
                let pixel = img.get_pixel(x as u32, y as u32);
                test_array[[x, y]] = pixel[0] as f64;
            }
        }

        assert_eq!(*array, test_array);
    }

    #[test]
    fn make_image_array_from_array2_u8() {
        let mut test_array = Array2::zeros((10, 5));
        test_array.mapv_inplace(|_| rand::random::<u8>());
        let array = ImageArray::from(&test_array);

        let test_array = test_array.map(|&x| x as f64);

        assert_eq!(*array, test_array);
    }

    #[test]
    fn make_image_array_from_array2_f64() {
        let mut test_array = Array2::zeros((10, 5));
        test_array.mapv_inplace(|_| rand::random::<f64>());

        let array = ImageArray::from(&test_array);

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
}
