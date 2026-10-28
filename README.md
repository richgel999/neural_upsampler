# jpegc_up2x: neural 2x upsampling as a drop-in C++ library

Note: this repo isn't JPEG specific; it's part of a larger JPEG focused project.

Four files, no dependencies beyond the standard library and the platform's thread API, no C++
exceptions or RTTI (it builds with both disabled), no heap allocation of its own (the worker threads' stacks come from the OS), C++17:

| file | what |
|---|---|
| `jpegc_up2x.h` | the API: one function, a model enum, a status enum; every caller caveat is in its comments |
| `jpegc_up2x.cpp` | the upsampler and the thread pool (`_beginthreadex` / `pthread_create`, status codes) |
| `jpegc_up2x_best_weights.h` | generated: the BEST model, 147 -> 64 -> 32 -> 12, 11948 weights (`jpegc_gen_header weights/up7w_all_fd.json lib_up2x/jpegc_up2x_best_weights.h JPEGC_UP2X_BEST` in the jpegc repo) |
| `jpegc_up2x_fast_weights.h` | generated: the FAST model, 147 -> 32 -> 16 -> 12, 5468 weights (from `weights/up7_fd.json`, prefix `JPEGC_UP2X_FAST`) |

Compile `jpegc_up2x.cpp` as C++17 and link the thread library on POSIX (`-pthread`). For output
byte-identical to the reference, do not let the compiler contract multiply-adds into FMAs
(MSVC `/fp:precise`, gcc / clang `-ffp-contract=off`); otherwise expect occasional 1-LSB differences.
The sibling library for JPEG artifact removal is `lib/` in the same repo; run that first on a
decoded JPEG, then this.

## Usage

```cpp
#include "jpegc_up2x.h"

// rgba: W x H, 4 bytes per pixel, R first, stride in bytes. out: at least 2H rows of out_stride bytes, 2W pixels each.
jpegc_up2x_status st = jpegc_up2x_rgba(rgba, W, H, W * 4,
                                       out, W * 8,
                                       JPEGC_UP2X_BEST,   // or JPEGC_UP2X_FAST: about twice the speed, 0.3 dB lower
                                       50,                // strength: 50 = as trained; 25 half the correction, 100 double
                                       0);                // threads: 0 = all hardware threads
if (st != JPEGC_UP2X_OK) { /* JPEGC_UP2X_BAD_ARGUMENT: nothing was written */ }
```

- The output cannot be in place (it is four times the size); the buffers must not overlap.
- Alpha is interpolated bilinearly with the same kernel as the base (straight alpha).
- For 4x, run the output through the function again; the second pass is slightly outside the
  training distribution but works well in practice.
- `threads` are created and joined inside the call; a thread that cannot start is not an error,
  its rows are done on the calling thread.

## What it is

Per input pixel, an MLP reads the 7x7 RGB window (147 inputs: the window as per-channel
differences from the centre, plus the centre's own RGB) and predicts the 2x2 output block as 12
residuals (4 pixels x RGB) over a bilinear interpolation of the same window; output pixel
(2x + i, 2y + j) sits a quarter of an input pixel from input pixel (x, y) toward its neighbour on
that side, the centre-aligned convention a 2x downsample inverts. Trained by finite differences on
RGB MSE, each training image the target of its own exact sRGB-aware Mitchell 2x downsample.

| model | net | weights | held-out RGB PSNR (148 images; bilinear 30.5, stb Catmull-Rom 31.0) | speed (32 threads) |
|---|---|---|---|---|
| BEST | 7x7, 64 / 32 | 11948 | 34.2 dB for the same recipe with a hold-out; the shipped weights trained on all 1390 images | 4.4 Mpx/s of input |
| FAST | 7x7, 32 / 16 | 5468 | 32.7 dB on the 52 photo images (BEST's photo-only ancestor: 33.0) | 9.9 Mpx/s of input |

## Verification

`jpegc_up2x_test` in the jpegc repo proves the library byte for byte against the trainer's
reference (`up_apply` + `up_compose` of `src/jc_up.cpp`) for both models: a real image with 1
thread and all threads, strengths 25 and 100, tiny synthetic images (1x1, 5x3, 7x100, 3 rows on 64
threads), row padding untouched, every bad argument refused with the output untouched, alpha equal
to a bilinear interpolation: 33 checks, passing on Windows (MSVC, Release and Debug) and Linux
(gcc, `-fno-exceptions -fno-rtti`), with identical output bytes across the two. The library's
output is also byte-identical to `jpegc_filter` with the same weights file. The code was reviewed
independently after its first version (no bugs found); its hardening suggestions are applied:
overlapping buffers refused, the system headers kept out of the anonymous namespace, static_asserts
on the weight counts and the base kind, and the documentation corrected on allocation (the worker
threads' stacks come from the OS), positive strides and clang's default FMA contraction.
