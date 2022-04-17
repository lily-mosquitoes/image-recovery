# Examples

You can run the examples here by calling `cargo run --example` with the example name, for *example* (:joy_cat:):

`cargo run --example denoise_each_channel` (`--debug` flag is implied)

or

`cargo run --release --example denoise_each_channel`

Have fun! :sparkles:

## Benchmarks

The source image for the example is a **400px by 400px** image.

The table below is *rough* benchmark of how fast each example runs on my particular machine/setup. This is here mostly so you know what to expect from running the examples.

|| `--debug` | `--release` |
|---|---|---|
`denoise_each_channel` | `~320s` | `~7s` |
`denoise_multichannel` | `~240s` | `~11s` |
