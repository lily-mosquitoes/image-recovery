use pretty_assertions::assert_eq;
use test::{Bencher, black_box};
use ndarray::{Array2, arr2};
use crate::utils;

static D_32: (usize, usize) = (32, 32);
static D_1024: (usize, usize) = (1024, 1024);

fn get_random_matrix(dimensions: (usize, usize)) -> Array2<f64> {
    let mut matrix = Array2::<f64>::zeros(dimensions);
    for x in 0..matrix.ncols() {
        for y in 0..matrix.nrows() {
            matrix[[x, y]] = rand::random::<u8>()  as f64;
        }
    }

    matrix
}

#[test]
fn len_of_vectors_is_correct() {
    let a = &get_random_matrix(D_32);
    let b = &get_random_matrix(D_32);

    let test = utils::len_of_vectors(a, b);
    let manual = ((a * a) + (b * b)).map(|x| x.sqrt());

    assert_eq!(test, manual);
}

#[test]
fn ball_projection_is_correct() {
    let a = &arr2(&[[3.0, -0.5], [-3.0, -0.5]]);
    let b = &arr2(&[[4.0, 0.5], [0.0, 0.5]]);

    let proj_a = arr2(&[[0.6, -0.5], [-1.0, -0.5]]);
    let proj_b = arr2(&[[0.8, 0.5], [0.0, 0.5]]);

    let test_proj = utils::ball_projection(a, b);

    assert_eq!(test_proj, (proj_a, proj_b));
}

#[bench]
fn bench_len_of_vectors(bench: &mut Bencher) {
    let a = &get_random_matrix(D_1024);
    let b = &get_random_matrix(D_1024);

    bench.iter(|| black_box(utils::len_of_vectors(a, b)));
}

#[bench]
fn bench_ball_projection(bench: &mut Bencher) {
    let a = &get_random_matrix(D_1024);
    let b = &get_random_matrix(D_1024);

    bench.iter(|| black_box(utils::ball_projection(a, b)));
}
