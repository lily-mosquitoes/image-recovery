# Examples

You can run the examples here by calling `cargo run --example` with the example name, for *example* (:joy_cat:):

`cargo run --example denoise_each_channel` (`--debug` flag is implied)

or

`cargo run --release --example denoise_each_channel`

Have fun! :sparkles:

## Benchmarks

The source image for the example is a **400px by 470px** image.

The table below is *rough* benchmark of how fast each example runs on my particular machine/setup. This is here mostly so you know what to expect from running the examples.

|| `--debug` | `--release`
|---|---|---|
`denoise_each_channel` | `~367s` | `~9s`
`denoise_multichannel` | `~280s` | `~13s`
