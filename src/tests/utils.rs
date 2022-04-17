// Copyright (C) 2022  Lílian Ferreira de Freitas & Emilia L. K. Blåsten
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published
// by the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

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

// TODO test multichannel variants

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
