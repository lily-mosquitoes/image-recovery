use image_recovery::{
    image,
    img::Manipulation,
    solvers,
};

fn main() {
    let img = image::open("examples/source_images/cute_birb_noisy.png")
        .expect("image could not be open")
        .into_rgb8();

    let img_matrices = img.to_matrices();

    // choice of inputs
    let norm_squared: f64 = 0.001; // 4.0; // remark after proof of theorem 3.1 (Chambolle, A. 2004. An Algorithm for Total Variation Minimization and Applications)

    let tau: f64 = 1.0 / 2_f64.sqrt(); //(1.0/norm_squared).sqrt();
    // let sigma: f64 = 2_f64.sqrt() / 8.0; // tau;
    let sigma: f64 = 1_f64 / (8.0 * tau);

    let lambda: f64 = 0.0000000000000001;

    let gamma: f64 = 0.35 * lambda;

    let denoised = solvers::denoise_multichannel(&img_matrices, lambda, tau, sigma, norm_squared, gamma);

    let new_img = image::RgbImage::from_matrices(&denoised);

    new_img.save("examples/result_images/cute_birb_denoised_multichannel.png")
        .expect("image could not be saved");
}
