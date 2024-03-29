use image_recovery::{
    image,      // re-exported `image` crate
    ImageArray, // struct for holding images
};

fn main() {
    // the `image` crate provides functionality to decode images
    let img = image::open("examples/source_images/angry_birb_noisy.png")
        .expect("image could not be open")
        .into_rgb8(); // the algorithms in this library are implemented for the Luma and Rgb
                      // types

    // transform the RGB image into a 3D Array
    let image_array = ImageArray::from(&img);

    // choose inputs for the denoising solver:
    // according to Chambolle, A. and Pock, T. (2011),
    // tau and lambda should be chosen such that
    // `tau * lambda * L2 norm^2 <= 1`
    // while `L2 norm^2 <= 8`
    // If we choose `tau * lambda * L2 norm^2 == 1`, then:
    let tau: f64 = 1.0 / 2_f64.sqrt();
    let sigma: f64 = 1_f64 / (8.0 * tau);

    // lambda drives the dual objective function
    // closer to zero results in a smoother output image
    // closer to infinity results in an output closer to the input
    let lambda: f64 = 0.0259624705;

    // gamma is a variable used to update the internal
    // state of the algorithm's variables, providing
    // an accelerated method for convergence.
    // Chambolle, A. and Pock, T. (2011), choose
    // the value to be `0.35 * lambda`
    let gamma: f64 = 0.35 * lambda;

    // choose bounds for denoising solver
    // the algorithm will run for at most `max_iter` iterations
    let max_iter: u32 = 500;

    // the algorithm will stop running if:
    // `convergence_threshold < norm(current - previous) / norm(previous)`
    // where `current` is the output candidate for the current iteration,
    // and `previous` is the output candidate of the previous iteration.
    let convergence_threshold = 10_f64.powi(-10);

    // now we can call the denoising solver with the chosen variables
    let denoised_array = image_array
        .denoise(lambda, tau, sigma, gamma, max_iter, convergence_threshold)
        .unwrap(); // will fail if image shape is 1 pixel in either x or y

    // we convert the solution into an RGB image format
    let denoised_img = denoised_array.into_rgb();

    // encode it and save it to a file
    denoised_img
        .save("examples/result_images/angry_birb_denoised.png")
        .expect("image could not be saved");
}
