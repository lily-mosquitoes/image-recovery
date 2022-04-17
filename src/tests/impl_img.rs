use pretty_assertions::assert_eq;
use test::{Bencher, black_box};
use ndarray::Array2;
use image::{RgbImage, Rgb};
use crate::{
    RgbMatrices,
    img::Manipulation,
};

static D_32: (usize, usize) = (32, 32);
static D_1024: (usize, usize) = (1024, 1024);

fn get_random_img_and_matrices(dimensions: (usize, usize)) -> (RgbImage, [Array2<f64>; 3]) {

    let mut img = RgbImage::new(dimensions.0 as u32, dimensions.1 as u32);
    let mut red = Array2::<f64>::zeros(dimensions);
    let mut green = Array2::<f64>::zeros(dimensions);
    let mut blue = Array2::<f64>::zeros(dimensions);

    for x in 0..dimensions.0 {
        for y in 0..dimensions.1 {
            let r = rand::random::<u8>();
            let g = rand::random::<u8>();
            let b = rand::random::<u8>();

            red[[x, y]] = r as f64;
            green[[x, y]] = g as f64;
            blue[[x, y]] = b as f64;

            img.put_pixel(x as u32, y as u32, Rgb([r, g, b]));
        }
    }

    (img, [red, green, blue])
}

#[test]
fn shape_is_correct() {
    let img = RgbImage::new(D_32.0 as u32, D_32.1 as u32);

    assert_eq!(D_32, img.shape())
}

#[test]
fn to_matrices_is_correct() {
    let (img, channels) = get_random_img_and_matrices(D_32);
    let img_matrices = img.to_matrices();

    assert_eq!(channels[0], img_matrices.red);
    assert_eq!(channels[1], img_matrices.green);
    assert_eq!(channels[2], img_matrices.blue);
}

#[test]
fn from_matrices_is_correct() {
    let (img, channels) = get_random_img_and_matrices(D_32);
    let img_matrices = RgbMatrices::from_channels(&channels[0], &channels[1], &channels[2]);
    let img_test = RgbImage::from_matrices(&img_matrices);

    assert_eq!(img, img_test);
}

#[bench]
fn bench_to_matrices(bench: &mut Bencher) {
    let (img, _) = get_random_img_and_matrices(D_1024);

    bench.iter(|| black_box(img.to_matrices()));
}

#[bench]
fn bench_from_matrices(bench: &mut Bencher) {
    let (_, channels) = get_random_img_and_matrices(D_1024);
    let img_matrices = RgbMatrices::from_channels(&channels[0], &channels[1], &channels[2]);

    bench.iter(|| black_box(RgbImage::from_matrices(&img_matrices)));
}
