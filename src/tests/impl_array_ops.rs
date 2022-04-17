use pretty_assertions::assert_eq;
use test::{Bencher, black_box};
use ndarray::Array2;
use crate::{
    array_ops::{Derivative, Power},
    RgbMatrices,
};

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

fn get_random_rbgmatrices(dimensions: (usize, usize)) -> RgbMatrices {
    let a = &get_random_matrix(dimensions);
    let b = &get_random_matrix(dimensions);
    let c = &get_random_matrix(dimensions);

    RgbMatrices::from_channels(a, b, c)
}

#[test]
fn dx_transposed_is_correct() {
    let a = &get_random_matrix(D_32);
    let b = &get_random_matrix(D_32);

    assert_eq!((a.dx() * b).sum(),
        (a * b.dx_transposed()).sum());
}

#[test]
fn dx_transposed_for_rgbmatrices_is_correct() {
    let a = &get_random_rbgmatrices(D_32);
    let b = &get_random_rbgmatrices(D_32);

    assert_eq!((a.dx() * b).sum(),
        (a * b.dx_transposed()).sum());
}

#[test]
fn dy_transposed_is_correct() {
    let a = &get_random_matrix(D_32);
    let b = &get_random_matrix(D_32);

    assert_eq!((a.dy() * b).sum(),
        (a * b.dy_transposed()).sum());
}

#[test]
fn dy_transposed_for_rgbmatrices_is_correct() {
    let a = &get_random_rbgmatrices(D_32);
    let b = &get_random_rbgmatrices(D_32);

    assert_eq!((a.dy() * b).sum(),
        (a * b.dy_transposed()).sum());
}

#[test]
fn squared_is_correct() {
    let a = &get_random_matrix(D_32);

    assert_eq!(a.squared(), a * a);
}

#[test]
fn squared_is_for_rgbmatrices_correct() {
    let a = &get_random_rbgmatrices(D_32);

    assert_eq!(a.squared(), a * a);
}

#[test]
fn powi_is_correct() {
    let a = &get_random_matrix(D_32);

    assert_eq!(a.powi(2), a * a);
}

#[test]
fn powi_for_rgbmatrices_is_correct() {
    let a = &get_random_rbgmatrices(D_32);

    assert_eq!(a.powi(2), a * a);
}

#[test]
fn powf_is_correct() {
    let a = &get_random_matrix(D_32);

    assert_eq!(a.powf(2.0), a * a);
}

#[test]
fn powf_for_rgbmatrices_is_correct() {
    let a = &get_random_rbgmatrices(D_32);

    assert_eq!(a.powf(2.0), a * a);
}

#[bench]
fn bench_dx(bench: &mut Bencher) {
    let a = &get_random_matrix(D_1024);

    bench.iter(|| black_box(a.dx()));
}

#[bench]
fn bench_rgbmatrices_dx(bench: &mut Bencher) {
    let a = &get_random_rbgmatrices(D_1024);

    bench.iter(|| black_box(a.dx()));
}

#[bench]
fn bench_dx_transposed(bench: &mut Bencher) {
    let a = &get_random_matrix(D_1024);

    bench.iter(|| black_box(a.dx_transposed()));
}

#[bench]
fn bench_rgbmatrices_dx_transposed(bench: &mut Bencher) {
    let a = &get_random_rbgmatrices(D_1024);

    bench.iter(|| black_box(a.dx_transposed()));
}

#[bench]
fn bench_dy(bench: &mut Bencher) {
    let a = &get_random_matrix(D_1024);

    bench.iter(|| black_box(a.dy()));
}

#[bench]
fn bench_rgbmatrices_dy(bench: &mut Bencher) {
    let a = &get_random_rbgmatrices(D_1024);

    bench.iter(|| black_box(a.dy()));
}

#[bench]
fn bench_dy_transposed(bench: &mut Bencher) {
    let a = &get_random_matrix(D_1024);

    bench.iter(|| black_box(a.dy_transposed()));
}

#[bench]
fn bench_rgbmatrices_dy_transposed(bench: &mut Bencher) {
    let a = &get_random_rbgmatrices(D_1024);

    bench.iter(|| black_box(a.dy_transposed()));
}

#[bench]
fn bench_squared(bench: &mut Bencher) {
    let a = &get_random_matrix(D_1024);

    bench.iter(|| black_box(a.squared()));
}

#[bench]
fn bench_rgbmatrices_squared(bench: &mut Bencher) {
    let a = &get_random_rbgmatrices(D_1024);

    bench.iter(|| black_box(a.squared()));
}

#[bench]
fn bench_powi(bench: &mut Bencher) {
    let a = &get_random_matrix(D_1024);

    bench.iter(|| black_box(a.powi(2)));
}

#[bench]
fn bench_rgbmatrices_powi(bench: &mut Bencher) {
    let a = &get_random_rbgmatrices(D_1024);

    bench.iter(|| black_box(a.powi(2)));
}

#[bench]
fn bench_powf(bench: &mut Bencher) {
    let a = &get_random_matrix(D_1024);

    bench.iter(|| black_box(a.powf(2.0)));
}

#[bench]
fn bench_rgbmatrices_powf(bench: &mut Bencher) {
    let a = &get_random_rbgmatrices(D_1024);

    bench.iter(|| black_box(a.powf(2.0)));
}
