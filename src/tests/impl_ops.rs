use pretty_assertions::assert_eq;
use test::{Bencher, black_box};
use ndarray::Array2;
use crate::RgbMatrices;

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

// TODO add tests for
// op<RgbMatrices>
// op<Array2<f64>>
// op<f64>

#[test]
fn mul_is_correct() {
    let a = &get_random_matrix(D_32);
    let b = &get_random_matrix(D_32);
    let c = &get_random_matrix(D_32);
    let ma = &RgbMatrices::from_channels(a, b, c);
    let mb = &RgbMatrices::from_channels(b, c, a);

    let x = &(a * b);
    let y = &(b * c);
    let z = &(c * a);

    assert_eq!(ma * mb,
        RgbMatrices::from_channels(x, y, z));
}

#[test]
fn div_is_correct() {
    let a = &get_random_matrix(D_32);
    let b = &get_random_matrix(D_32);
    let c = &get_random_matrix(D_32);
    let ma = &RgbMatrices::from_channels(a, b, c);
    let mb = &RgbMatrices::from_channels(b, c, a);

    let x = &(a / b);
    let y = &(b / c);
    let z = &(c / a);

    assert_eq!(ma / mb,
        RgbMatrices::from_channels(x, y, z));
}

#[test]
fn add_is_correct() {
    let a = &get_random_matrix(D_32);
    let b = &get_random_matrix(D_32);
    let c = &get_random_matrix(D_32);
    let ma = &RgbMatrices::from_channels(a, b, c);
    let mb = &RgbMatrices::from_channels(b, c, a);

    let x = &(a + b);
    let y = &(b + c);
    let z = &(c + a);

    assert_eq!(ma + mb,
        RgbMatrices::from_channels(x, y, z));
}

#[test]
fn sub_is_correct() {
    let a = &get_random_matrix(D_32);
    let b = &get_random_matrix(D_32);
    let c = &get_random_matrix(D_32);
    let ma = &RgbMatrices::from_channels(a, b, c);
    let mb = &RgbMatrices::from_channels(b, c, a);

    let x = &(a - b);
    let y = &(b - c);
    let z = &(c - a);

    assert_eq!(ma - mb,
        RgbMatrices::from_channels(x, y, z));
}

#[bench]
fn bench_mul(bench: &mut Bencher) {
    let a = &get_random_matrix(D_1024);
    let b = &get_random_matrix(D_1024);
    let c = &get_random_matrix(D_1024);
    let ma = &RgbMatrices::from_channels(a, b, c);
    let mb = &RgbMatrices::from_channels(b, c, a);

    bench.iter(|| black_box(ma * mb));
}

#[bench]
fn bench_div(bench: &mut Bencher) {
    let a = &get_random_matrix(D_1024);
    let b = &get_random_matrix(D_1024);
    let c = &get_random_matrix(D_1024);
    let ma = &RgbMatrices::from_channels(a, b, c);
    let mb = &RgbMatrices::from_channels(b, c, a);

    bench.iter(|| black_box(ma / mb));
}

#[bench]
fn bench_add(bench: &mut Bencher) {
    let a = &get_random_matrix(D_1024);
    let b = &get_random_matrix(D_1024);
    let c = &get_random_matrix(D_1024);
    let ma = &RgbMatrices::from_channels(a, b, c);
    let mb = &RgbMatrices::from_channels(b, c, a);

    bench.iter(|| black_box(ma + mb));
}

#[bench]
fn bench_sub(bench: &mut Bencher) {
    let a = &get_random_matrix(D_1024);
    let b = &get_random_matrix(D_1024);
    let c = &get_random_matrix(D_1024);
    let ma = &RgbMatrices::from_channels(a, b, c);
    let mb = &RgbMatrices::from_channels(b, c, a);

    bench.iter(|| black_box(ma - mb));
}
