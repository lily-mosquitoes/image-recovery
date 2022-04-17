use pretty_assertions::assert_eq;
// use test::{Bencher, black_box}; // TODO bench
use ndarray::Array2;
use crate::array_ops::Derivative;

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

// TODO test other helper functions
// TODO test multichannel variants

#[test]
fn k_star_is_correct() {
    let a = &get_random_matrix(D_32);
    let b1 = &get_random_matrix(D_32);
    let b2 = &get_random_matrix(D_32);

    let k_star = |a: &Array2<f64>, b: &Array2<f64>| a.dx_transposed() + b.dy_transposed();

    let manual = (a.dx() * b1).sum() + (a.dy() * b2).sum();

    let test = (a * k_star(b1, b2)).sum();

    assert_eq!(manual, test);
}
