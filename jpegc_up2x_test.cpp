// jpegc_up2x_test: the standalone upsampler library against the trainer's reference.
//   jpegc_up2x_test IN.png [OUT.png]
// Loads the image (RGB), builds an RGBA copy with a non-trivial alpha and stride, upsamples it
// through jpegc_up2x_rgba() with both models, 1 thread and all threads, and compares every RGB
// byte against up_apply() + up_compose() of src/jc_up.cpp with the same weights file, and the
// alpha against a bilinear interpolation. Then tiny synthetic images, bad arguments, strength.
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include "jpegc_up2x.h"
#include "jc_image.h"
#include "jc_mlp.h"
#include "jc_noise.h"
#include "jc_up.h"

static int g_fails = 0;
static void check(bool ok, const char* what) { printf("  %-56s %s\n", what, ok ? "PASS" : "FAIL"); g_fails += !ok; }

struct Ref { UpDesc d; MLP m; };
static bool load_ref(const char* path, Ref& r) { std::string t, why; if (!up_read_weights_json(path, r.d, r.m, t, why)) { fprintf(stderr, "%s\n", why.c_str()); return false; } return true; }

// The reference 2x of an RGB image at strength s (rscale s / 50).
static void reference(const Ref& ref, const Image8& lr, int strength, Image8& out) {
    std::vector<float> r;
    up_apply(ref.d, ref.m, lr, r, 1);
    up_compose(ref.d, lr, r, (float)strength / 50.0f, out);
}
// The reference alpha: the bilinear base on the alpha plane alone (up_base on a gray-as-RGB image).
static void reference_alpha(const Ref& ref, const Image8& lr_alpha_as_rgb, Image8& out) { up_compose(ref.d, lr_alpha_as_rgb, std::vector<float>(), 0.0f, out); }

static std::vector<uint8_t> rgba_of(const Image8& rgb, int pad, int& stride) {
    stride = rgb.w * 4 + pad;
    std::vector<uint8_t> v((size_t)stride * rgb.h, 0xEE);
    for (int y = 0; y < rgb.h; y++) for (int x = 0; x < rgb.w; x++) {
        const uint8_t* p = rgb.at(x, y); uint8_t* o = &v[(size_t)y * stride + (size_t)x * 4];
        o[0] = p[0]; o[1] = p[1]; o[2] = p[2]; o[3] = (uint8_t)(x * 7 + y * 13);
    }
    return v;
}
// Alpha of the input as an RGB image (for the alpha reference).
static Image8 alpha_image(const Image8& rgb) {
    Image8 a; a.resize(rgb.w, rgb.h);
    for (int y = 0; y < rgb.h; y++) for (int x = 0; x < rgb.w; x++) { uint8_t* o = a.at(x, y); o[0] = o[1] = o[2] = (uint8_t)(x * 7 + y * 13); }
    return a;
}
// Output pixels (2W x 2H in `buf`, stride) differing from ref in RGB, and alpha bytes differing from ref_alpha's channel 0.
static size_t diff_of(const std::vector<uint8_t>& buf, int stride, const Image8& ref, const Image8& ref_alpha, size_t& bad_alpha) {
    size_t bad = 0; bad_alpha = 0;
    for (int y = 0; y < ref.h; y++) for (int x = 0; x < ref.w; x++) {
        const uint8_t* e = ref.at(x, y); const uint8_t* o = &buf[(size_t)y * stride + (size_t)x * 4];
        bad += o[0] != e[0] || o[1] != e[1] || o[2] != e[2];
        bad_alpha += o[3] != ref_alpha.at(x, y)[0];
    }
    return bad;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "jpegc_up2x_test IN.png [OUT.png]\n"); return 1; }
    const std::string out_path = argc > 2 ? argv[2] : "";
    std::string why;
    Image8 rgb;
    if (!load_image_rgb(argv[1], rgb, why)) { fprintf(stderr, "%s\n", why.c_str()); return 1; }
    Ref refs[2];
    if (!load_ref("weights/up7w_all_fd.json", refs[0]) || !load_ref("weights/up7_fd.json", refs[1])) return 1;
    const jpegc_up2x_model models[2] = { JPEGC_UP2X_BEST, JPEGC_UP2X_FAST };
    const char* names[2] = { "BEST", "FAST" };
    check(jpegc_up2x_weights(JPEGC_UP2X_BEST) == (int)refs[0].m.p.size() && jpegc_up2x_weights(JPEGC_UP2X_FAST) == (int)refs[1].m.p.size(), "weight counts match the weights files");

    // ---- the image: both models, 1 thread and all threads
    const int W = rgb.w, H = rgb.h;
    int in_stride; std::vector<uint8_t> in = rgba_of(rgb, 12, in_stride);
    const int out_stride = W * 8 + 4;
    std::vector<uint8_t> out((size_t)out_stride * 2 * H, 0x11);
    const Image8 alpha_in = alpha_image(rgb);
    for (int mi = 0; mi < 2; mi++) {
        Image8 ref, ref_a; reference(refs[mi], rgb, 50, ref); reference_alpha(refs[mi], alpha_in, ref_a);
        for (int threads : { 1, 0 }) {
            const auto t0 = std::chrono::steady_clock::now();
            const bool ok = JPEGC_UP2X_OK == jpegc_up2x_rgba(in.data(), W, H, in_stride, out.data(), out_stride, models[mi], 50, threads);
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            size_t bad_a = 0; const size_t bad = ok ? diff_of(out, out_stride, ref, ref_a, bad_a) : (size_t)W * H * 4;
            char what[128]; snprintf(what, sizeof(what), "%s %dx%d, threads %d: %zu px differ, %zu alpha, %.1f ms (%.1f Mpx/s in)", names[mi], W, H, threads, bad, bad_a, ms, (double)W * H / ms * 1e-3);
            check(ok && bad == 0 && bad_a == 0, what);
        }
        if (mi == 0 && !out_path.empty()) {
            Image8 o; o.resize(2 * W, 2 * H);
            for (int y = 0; y < 2 * H; y++) for (int x = 0; x < 2 * W; x++) memcpy(o.at(x, y), &out[(size_t)y * out_stride + (size_t)x * 4], 3);
            if (!save_png(out_path, o, why)) { fprintf(stderr, "%s\n", why.c_str()); return 1; }
            printf("output  : %s (%dx%d, BEST)\n", out_path.c_str(), o.w, o.h);
        }
    }
    size_t pad_bad = 0;
    for (int y = 0; y < 2 * H; y++) for (int k = W * 8; k < out_stride; k++) pad_bad += out[(size_t)y * out_stride + k] != 0x11;
    check(pad_bad == 0, "row padding of the output untouched");

    // ---- strength 100 and 25 against the reference's rscale
    for (int s : { 100, 25 }) {
        Image8 ref, ref_a; reference(refs[0], rgb, s, ref); reference_alpha(refs[0], alpha_in, ref_a);
        const bool ok = JPEGC_UP2X_OK == jpegc_up2x_rgba(in.data(), W, H, in_stride, out.data(), out_stride, JPEGC_UP2X_BEST, s, 0);
        size_t bad_a = 0; const size_t bad = ok ? diff_of(out, out_stride, ref, ref_a, bad_a) : 1;
        char what[64]; snprintf(what, sizeof(what), "BEST at strength %d", s);
        check(ok && bad == 0 && bad_a == 0, what);
    }

    // ---- tiny synthetic images, both models, 1 thread and 64 threads
    {
        const int sizes[][2] = { { 1, 1 }, { 5, 3 }, { 7, 100 }, { 100, 3 }, { 9, 9 }, { 17, 40 } };
        for (const auto& sz : sizes) {
            Image8 img; img.resize(sz[0], sz[1]);
            for (size_t i = 0; i < img.rgb.size(); i++) img.rgb[i] = (uint8_t)(jc_noise_u32(7, 1, (uint32_t)sz[0], (uint32_t)sz[1], i) >> 24);
            const Image8 a_in = alpha_image(img);
            for (int mi = 0; mi < 2; mi++) {
                Image8 ref, ref_a; reference(refs[mi], img, 50, ref); reference_alpha(refs[mi], a_in, ref_a);
                for (int threads : { 1, 64 }) {
                    int st; std::vector<uint8_t> a = rgba_of(img, 8, st);
                    const int ost = img.w * 8;
                    std::vector<uint8_t> o((size_t)ost * 2 * img.h, 0x33);
                    const bool ok = JPEGC_UP2X_OK == jpegc_up2x_rgba(a.data(), img.w, img.h, st, o.data(), ost, models[mi], 50, threads);
                    size_t bad_a = 0; const size_t bad = ok ? diff_of(o, ost, ref, ref_a, bad_a) : 1;
                    char what[96]; snprintf(what, sizeof(what), "%s %dx%d on %d threads", names[mi], img.w, img.h, threads);
                    check(ok && bad == 0 && bad_a == 0, what);
                }
            }
        }
    }
    // ---- bad arguments: refused, the output untouched
    {
        std::vector<uint8_t> o((size_t)out_stride * 2 * H, 0x55), o0 = o;
        bool any = false;
        any |= JPEGC_UP2X_OK == jpegc_up2x_rgba(in.data(), 0x7FFFFFFF, 1, 0x7FFFFFFF, o.data(), 0x7FFFFFFF, JPEGC_UP2X_BEST, 50, 1);
        any |= JPEGC_UP2X_OK == jpegc_up2x_rgba(o.data(), W / 2, H / 2, out_stride, o.data() + out_stride, out_stride, JPEGC_UP2X_BEST, 50, 1);   // overlapping buffers|= JPEGC_UP2X_OK == jpegc_up2x_rgba(nullptr, W, H, in_stride, o.data(), out_stride, JPEGC_UP2X_BEST, 50, 1);
        any |= JPEGC_UP2X_OK == jpegc_up2x_rgba(in.data(), W, H, in_stride, nullptr, out_stride, JPEGC_UP2X_BEST, 50, 1);
        any |= JPEGC_UP2X_OK == jpegc_up2x_rgba(in.data(), 0, H, in_stride, o.data(), out_stride, JPEGC_UP2X_BEST, 50, 1);
        any |= JPEGC_UP2X_OK == jpegc_up2x_rgba(in.data(), W, -1, in_stride, o.data(), out_stride, JPEGC_UP2X_BEST, 50, 1);
        any |= JPEGC_UP2X_OK == jpegc_up2x_rgba(in.data(), W, H, W * 4 - 1, o.data(), out_stride, JPEGC_UP2X_BEST, 50, 1);
        any |= JPEGC_UP2X_OK == jpegc_up2x_rgba(in.data(), W, H, in_stride, o.data(), W * 8 - 4, JPEGC_UP2X_BEST, 50, 1);   // the output needs 2W pixels per row
        any |= JPEGC_UP2X_OK == jpegc_up2x_rgba(in.data(), W, H, in_stride, o.data(), out_stride, (jpegc_up2x_model)2, 50, 1);
        any |= JPEGC_UP2X_OK == jpegc_up2x_rgba(in.data(), W, H, in_stride, o.data(), out_stride, JPEGC_UP2X_BEST, 0, 1);
        any |= JPEGC_UP2X_OK == jpegc_up2x_rgba(in.data(), W, H, in_stride, o.data(), out_stride, JPEGC_UP2X_BEST, 301, 1);
        any |= JPEGC_UP2X_OK == jpegc_up2x_rgba(in.data(), 0x7FFFFFFF, 1, 0x7FFFFFFF, o.data(), 0x7FFFFFFF, JPEGC_UP2X_BEST, 50, 1);
        check(!any && o == o0, "bad arguments refused, output untouched");
    }
    printf(g_fails ? "test: %d failed\n" : "test: all passed\n", g_fails);
    return g_fails ? 1 : 0;
}
