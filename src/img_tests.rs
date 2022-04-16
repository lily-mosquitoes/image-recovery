use pretty_assertions::assert_eq;
use test::{Bencher, black_box};
use ndarray::Array2;
use image::{RgbImage, Rgb};
use crate::img::{Manipulation, Channel};

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
fn get_channel_is_correct() {
    let (img, channels) = get_random_img_and_matrices(D_32);

    let red = img.get_channel(Channel::Red);
    assert_eq!(red, channels[0]);

    let green = img.get_channel(Channel::Green);
    assert_eq!(green, channels[1]);

    let blue = img.get_channel(Channel::Blue);
    assert_eq!(blue, channels[2]);
}

#[test]
fn update_channel_is_correct() {
    let (img, channels) = get_random_img_and_matrices(D_32);

    let mut new_img = RgbImage::new(D_32.0 as u32, D_32.1 as u32);

    new_img.update_channel(Channel::Red, &channels[0]).unwrap();
    new_img.update_channel(Channel::Green, &channels[1]).unwrap();
    new_img.update_channel(Channel::Blue, &channels[2]).unwrap();

    assert_eq!(new_img, img);
}

#[test]
#[should_panic(expected = "incorrect dimensions")]
fn update_channel_panics_on_wrong_dimensions() {
    let (_, channels) = get_random_img_and_matrices((20, 20));

    let mut new_img = RgbImage::new(10, 10);

    new_img.update_channel(Channel::Red, &channels[0])
        .expect("incorrect dimensions");
}

#[bench]
fn bench_get_channel(bench: &mut Bencher) {
    let (img, _) = get_random_img_and_matrices(D_1024);

    bench.iter(|| black_box(img.get_channel(Channel::Green)));
}

#[bench]
fn bench_update_channel(bench: &mut Bencher) {
    let (_, channels) = get_random_img_and_matrices(D_1024);
    let mut new_img = RgbImage::new(D_1024.0 as u32, D_1024.1 as u32);

    bench.iter(|| black_box(new_img.update_channel(Channel::Green, &channels[1]).unwrap()));
}
