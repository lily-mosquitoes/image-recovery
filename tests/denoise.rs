use image_recovery::{
    RgbMatrices,
    img::Manipulation,
    solvers,
};

#[test]
fn denoise_example() {
    let img = image::open("dev_images/birb_noisy.png")
        .expect("image could not be open")
        .into_rgb8();
    // let d = img.dimensions();

    let img_matrices = img.to_matrices();

    // choice of inputs
    let norm_squared: f64 = 0.001; // 4.0; // remark after proof of theorem 3.1 (Chambolle, A. 2004. An Algorithm for Total Variation Minimization and Applications)

    let tau: f64 = 1.0 / 2_f64.sqrt(); //(1.0/norm_squared).sqrt();
    // let sigma: f64 = 2_f64.sqrt() / 8.0; // tau;
    let sigma: f64 = 1_f64 / (8.0 * tau);

    let lambda: f64 = 0.0000000000000001;

    let gamma: f64 = 0.35 * lambda;

    let denoised_red = solvers::denoise(&img_matrices.red, lambda, tau, sigma, norm_squared, gamma);
    let denoised_green = solvers::denoise(&img_matrices.green, lambda, tau, sigma, norm_squared, gamma);
    let denoised_blue = solvers::denoise(&img_matrices.blue, lambda, tau, sigma, norm_squared, gamma);

    let denoised = RgbMatrices::from_channels(&denoised_red, &denoised_green, &denoised_blue);

    let new_img = image::RgbImage::from_matrices(&denoised);

    new_img.save("dev_images/birb_denoise_each_channel.png")
        .expect("image could not be saved");
}

#[test]
fn denoise_multichannel_example() {
    let img = image::open("dev_images/birb_noisy.png")
        .expect("image could not be open")
        .into_rgb8();
    // let d = img.dimensions();

    let img_matrices = img.to_matrices();

    // let mut red_img = image::RgbImage::new(d.0, d.1);
    // red_img.update_channel(Channel::Red, &red_channel)
    //     .expect("unexpected image size");
    //
    // red_img.save("dev_images/birb_red.png")
    //     .expect("image could not be saved");

    // choice of inputs
    let norm_squared: f64 = 0.001; // 4.0; // remark after proof of theorem 3.1 (Chambolle, A. 2004. An Algorithm for Total Variation Minimization and Applications)

    let tau: f64 = 1.0 / 2_f64.sqrt(); //(1.0/norm_squared).sqrt();
    // let sigma: f64 = 2_f64.sqrt() / 8.0; // tau;
    let sigma: f64 = 1_f64 / (8.0 * tau);

    let lambda: f64 = 0.0000000000000001;

    let gamma: f64 = 0.35 * lambda;

    let denoised = solvers::denoise_multichannel(&img_matrices, lambda, tau, sigma, norm_squared, gamma);

    let new_img = image::RgbImage::from_matrices(&denoised);

    new_img.save("dev_images/birb_denoise_multichannel.png")
        .expect("image could not be saved");
}
