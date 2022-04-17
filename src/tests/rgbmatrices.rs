use pretty_assertions::assert_eq;
// use test::{Bencher, black_box}; // TODO bench
use ndarray::Array2;
use crate::RgbMatrices;

static D_32: (usize, usize) = (32, 32);
// static D_1024: (usize, usize) = (1024, 1024); // TODO bench

fn get_random_matrix(dimensions: (usize, usize)) -> Array2<f64> {
    let mut matrix = Array2::<f64>::zeros(dimensions);
    for x in 0..matrix.ncols() {
        for y in 0..matrix.nrows() {
            matrix[[x, y]] = rand::random::<u8>()  as f64;
        }
    }

    matrix
}

// TODO map is correct

#[test]
fn new_is_correct() {
    let zeroes = Array2::<f64>::zeros(D_32);
    let img_matrices = RgbMatrices::new(D_32);

    assert_eq!(zeroes, img_matrices.red);
    assert_eq!(zeroes, img_matrices.green);
    assert_eq!(zeroes, img_matrices.blue);
}

#[test]
fn from_channels_is_correct() {
    let red = &get_random_matrix(D_32);
    let green = &get_random_matrix(D_32);
    let blue = &get_random_matrix(D_32);

    let img_matrices = RgbMatrices::from_channels(red, green, blue);

    assert_eq!(red, img_matrices.red);
    assert_eq!(green, img_matrices.green);
    assert_eq!(blue, img_matrices.blue);
}

#[test]
fn sum_is_correct() {
    let red = &get_random_matrix(D_32);
    let green = &get_random_matrix(D_32);
    let blue = &get_random_matrix(D_32);
    let sum = red.sum() + green.sum() + blue.sum();

    let img_matrices = RgbMatrices::from_channels(red, green, blue);

    assert_eq!(sum, img_matrices.sum());
}
