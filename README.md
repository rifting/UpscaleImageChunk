# UpscaleImageChunk
\- now with GPU!!!!

A project I made to upscale low quality crops of images, provided you have the original.

## Using

```
Usage: upscale_image_chunk.exe [OPTIONS] --cropped-image <CROPPED_IMAGE> --full-image <FULL_IMAGE> --min-length <MIN_LENGTH> --stop-at <STOP_AT>

Options:
  -c, --cropped-image <CROPPED_IMAGE>  Cropped image / Chunk from the original image
  -f, --full-image <FULL_IMAGE>        Full image
  -o, --output <OUTPUT>                Output file [default: out.png]
  -m, --min-length <MIN_LENGTH>        Minimum size of the cropped image's longest edge. UIC will resize the image before comparisons
  -s, --stop-at <STOP_AT>              Stop comparisons once the cropped chunk's longest edge is this size
  -h, --help                           Print help
  -V, --version                        Print version
```

## Compiling
- Ensure you have CUDA installed

- `nvcc --ptx ssd_kernel.cu -o ssd_kernel.ptx`

- `cp ssd_kernel.ptx src`

- `cargo build --release`

## So how does it work?

If you've ever messed around in photo editing software, you might've noticed when you overlay an image with an inverted copy of itself, and make it 50% opaque, the photo becomes entirely gray. The same concept is applied here, only we don't know where to overlay the image. 

In order to find that area and size we need to crop from the larger image, we take the smaller cropped image and slide it over every possible position it could be in the full image and measure how different the cropped image is from that section of the full image. We keep track of the position and scale of the image when it's in an area with the least differences. If the image is ever overlayed somewhere with even less differences, we clear out the old location and size and replace it with the new ones.

Once we've slid the image over every possible area we can, we make it 1 pixel wider/taller and do it all over again, until we hit a number that would be unrealistic for the image to be as big as. Then we just look back and see what coordinates and scale had the highest similarity to the full image, cut that chunk out of the full image, and save it to disk!