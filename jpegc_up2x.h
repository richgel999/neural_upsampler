// jpegc_up2x: a neural 2x upsampler for RGBA images in memory, as one C++17 source file plus two
// generated weights headers (one per model). No dependencies beyond the standard library and the
// platform's thread API; no C++ exceptions are thrown, caught or required (it builds with
// exceptions disabled), no RTTI; every failure is a status code.
//
// Per input pixel a small MLP reads the 7x7 RGB window around it (as differences from the centre,
// plus the centre's own RGB) and predicts the 2x2 block of output pixels as RGB corrections over
// a bilinear interpolation of the same window. The output pixel (2x + i, 2y + j) sits a quarter
// of an input pixel from the centre of input pixel (x, y), toward its neighbour on that side
// (the usual centre-aligned 2x convention, the one a 2x downsample inverts). Trained by
// jpegc_train --mode up2x on photographs and web images, each the target of its own exact
// sRGB-aware Mitchell 2x downsample, by RGB MSE (see the jpegc README).
//
// What the caller must know:
//   - The net reads and writes plain sRGB values; alpha is interpolated bilinearly with the same
//     kernel as the base (straight, non-premultiplied alpha).
//   - The output is 2 * width x 2 * height in a caller-supplied buffer; it cannot be in place.
//   - The models were trained on photographic and web content at the sizes they are shown; on
//     synthetic graphics, text or noise the correction can overshoot at hard edges (strength
//     below 50 tames it). Running the output through the upsampler again gives 4x; the second
//     pass is slightly outside the training distribution but works well in practice.
//   - The output is byte-identical to the reference only when the compiler does not contract
//     multiply-adds into FMAs (MSVC /fp:precise; gcc and clang need -ffp-contract=off explicitly,
//     clang contracts by default); with FMA contraction or fast-math the output can differ by 1 LSB
//     here and there.
#pragma once
#include <stddef.h>
#include <stdint.h>

enum jpegc_up2x_model {
    JPEGC_UP2X_BEST = 0,   // 147 -> 64 -> 32 -> 12, 11948 weights, trained on 1390 images: the best quality
    JPEGC_UP2X_FAST = 1    // 147 -> 32 -> 16 -> 12, 5468 weights: about twice the speed; on held-out photos 32.7 dB against 33.0 for
                           // a BEST-shaped net trained on the same corpus (the shipped BEST, trained on more, measures 34.2 on a wider set)
};

enum jpegc_up2x_status {
    JPEGC_UP2X_OK = 0,
    JPEGC_UP2X_BAD_ARGUMENT = 1   // a null pointer, a bad size, stride, model, strength; nothing written
};

// Upsamples a W x H RGBA image (4 bytes per pixel, R first in memory) 2x into `out`, a buffer of
// at least 2H rows of out_stride bytes, each holding 2W pixels. Strides are positive, in bytes;
// the buffers must not overlap (an overlap is refused as a bad argument).
//   model     JPEGC_UP2X_BEST or JPEGC_UP2X_FAST
//   strength  50 = the net as trained; 25 halves the correction, 100 doubles it (1..300); the
//             bilinear base is always applied in full
//   threads   worker threads for the pass (0 or less = the hardware thread count; 1 = the
//             calling thread only), created with the platform's thread API and joined inside the
//             call. A thread that cannot be created is not an error: its rows are done on the
//             calling thread, so the call always completes the whole image when it returns OK.
// The library itself allocates nothing (with threads != 1 the OS / C runtime allocates the worker
// threads' stacks). Returns JPEGC_UP2X_OK, or a status with `out` untouched.
jpegc_up2x_status jpegc_up2x_rgba(const uint8_t* in, int width, int height, int in_stride,
                                  uint8_t* out, int out_stride, jpegc_up2x_model model, int strength, int threads);

// The number of weights of a model (0 for an unknown model), for reporting.
int jpegc_up2x_weights(jpegc_up2x_model model);
