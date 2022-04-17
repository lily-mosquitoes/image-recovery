# Examples

You can run the examples here by calling `cargo run --example` with the example name, for *example* (:joy_cat:):

`cargo run --example denoise`

Have fun! :sparkles:

## Benchmarks

The source image for the example is a **400px by 400px** image.

The table below is loose benchmark of how fast each example runs on my particular machine/setup.

|| `--debug` | `--release` |
|---|---|---|
`denoise` | `~320s` | |
`denoise_multichannel` | `~240s` | |
