// jpegc_up2x.cpp: see jpegc_up2x.h. The arithmetic mirrors the trainer's reference
// (src/jc_up.cpp up_features() / up_base() / up_apply() / up_compose(), src/jc_mlp.h
// mlp_forward_linear()) expression for expression, so the output is byte-identical to
// jpegc_filter's with the same weights file.
#include "jpegc_up2x.h"
#include "jpegc_up2x_best_weights.h"
#include "jpegc_up2x_fast_weights.h"
#include <algorithm>
#include <cmath>
// The platform thread API (outside the anonymous namespace below: system headers must not be declared inside it).
#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <process.h>
#else
#include <pthread.h>
#include <unistd.h>
#endif

namespace {

// ---------------------------------------------------------------- the models
struct Model {
    int radius, in_abs, base, nin, nout, nlayers, nweights;
    float abs_gain, leak;
    const int* hidden;
    const float* weights;
};
const Model kBest = { JPEGC_UP2X_BEST_WIN, JPEGC_UP2X_BEST_IN_ABS, JPEGC_UP2X_BEST_BASE, JPEGC_UP2X_BEST_NIN, JPEGC_UP2X_BEST_NOUT, JPEGC_UP2X_BEST_NLAYERS,
                      JPEGC_UP2X_BEST_NWEIGHTS, JPEGC_UP2X_BEST_ABS_GAIN, JPEGC_UP2X_BEST_LEAK, JPEGC_UP2X_BEST_hidden, JPEGC_UP2X_BEST_weights };
const Model kFast = { JPEGC_UP2X_FAST_WIN, JPEGC_UP2X_FAST_IN_ABS, JPEGC_UP2X_FAST_BASE, JPEGC_UP2X_FAST_NIN, JPEGC_UP2X_FAST_NOUT, JPEGC_UP2X_FAST_NLAYERS,
                      JPEGC_UP2X_FAST_NWEIGHTS, JPEGC_UP2X_FAST_ABS_GAIN, JPEGC_UP2X_FAST_LEAK, JPEGC_UP2X_FAST_hidden, JPEGC_UP2X_FAST_weights };
const int kMaxR = 5, kMaxNin = ((2 * kMaxR + 1) * (2 * kMaxR + 1) - 1) * 3 + 3, kNout = 12;
const int kMaxW = JPEGC_UP2X_BEST_MAXW > JPEGC_UP2X_FAST_MAXW ? JPEGC_UP2X_BEST_MAXW : JPEGC_UP2X_FAST_MAXW;
static_assert(JPEGC_UP2X_BEST_NIN == ((2 * JPEGC_UP2X_BEST_WIN + 1) * (2 * JPEGC_UP2X_BEST_WIN + 1) - 1) * 3 + (JPEGC_UP2X_BEST_IN_ABS ? 3 : 0), "BEST: nin does not match its window");
static_assert(JPEGC_UP2X_FAST_NIN == ((2 * JPEGC_UP2X_FAST_WIN + 1) * (2 * JPEGC_UP2X_FAST_WIN + 1) - 1) * 3 + (JPEGC_UP2X_FAST_IN_ABS ? 3 : 0), "FAST: nin does not match its window");
static_assert(JPEGC_UP2X_BEST_NOUT == 12 && JPEGC_UP2X_FAST_NOUT == 12, "the models predict a 2x2 block x RGB");
static_assert(JPEGC_UP2X_BEST_WIN <= kMaxR && JPEGC_UP2X_FAST_WIN <= kMaxR, "the feature buffer holds windows up to 11x11");
static_assert(kMaxW >= 1 && kMaxW <= 256, "the activation buffers hold hidden widths of 1..256");
static_assert(JPEGC_UP2X_BEST_BASE == 1 || JPEGC_UP2X_BEST_BASE == 2, "BEST: base must be bilinear (1) or Catmull-Rom (2)");
static_assert(JPEGC_UP2X_FAST_BASE == 1 || JPEGC_UP2X_FAST_BASE == 2, "FAST: base must be bilinear (1) or Catmull-Rom (2)");
// the weight arrays hold exactly what the shapes need (an array literal with too few initialisers would zero-fill silently)
constexpr int count2(int nin, int h0, int h1, int nout) { return nin * h0 + h0 + h0 * h1 + h1 + h1 * nout + nout; }
static_assert(JPEGC_UP2X_BEST_NLAYERS == 2 && JPEGC_UP2X_FAST_NLAYERS == 2, "the count check below assumes two hidden layers");
static_assert(count2(JPEGC_UP2X_BEST_NIN, JPEGC_UP2X_BEST_HIDDEN_0, JPEGC_UP2X_BEST_HIDDEN_1, JPEGC_UP2X_BEST_NOUT) == JPEGC_UP2X_BEST_NWEIGHTS, "BEST: weight count");
static_assert(count2(JPEGC_UP2X_FAST_NIN, JPEGC_UP2X_FAST_HIDDEN_0, JPEGC_UP2X_FAST_HIDDEN_1, JPEGC_UP2X_FAST_NOUT) == JPEGC_UP2X_FAST_NWEIGHTS, "FAST: weight count");
static_assert(JPEGC_UP2X_BEST_HIDDEN_0 <= kMaxW && JPEGC_UP2X_BEST_HIDDEN_1 <= kMaxW && JPEGC_UP2X_FAST_HIDDEN_0 <= kMaxW && JPEGC_UP2X_FAST_HIDDEN_1 <= kMaxW, "the activation buffers hold every hidden layer");

const float kInv255 = 1.0f / 255.0f;
inline float u8f(uint8_t v) { return (float)v * kInv255; }
inline int clampi(int v, int lo, int hi) { return v < lo ? lo : v > hi ? hi : v; }

// = up_features() of jc_up.cpp, on an RGBA row stride: the window as per-channel differences from
// the centre (the centre's own slots dropped), then the centre's absolute RGB.
void features(const Model& m, const uint8_t* in, int stride, int W, int H, int px, int py, float* f) {
    const uint8_t* c = in + (size_t)py * stride + (size_t)px * 4;
    const float c0 = u8f(c[0]), c1 = u8f(c[1]), c2 = u8f(c[2]);
    int k = 0;
    const int R = m.radius;
    for (int dy = -R; dy <= R; dy++) {
        const int yy = clampi(py + dy, 0, H - 1);
        const uint8_t* row = in + (size_t)yy * stride;
        for (int dx = -R; dx <= R; dx++) {
            if (dx == 0 && dy == 0) continue;
            const uint8_t* q = row + (size_t)clampi(px + dx, 0, W - 1) * 4;
            f[k++] = u8f(q[0]) - c0;
            f[k++] = u8f(q[1]) - c1;
            f[k++] = u8f(q[2]) - c2;
        }
    }
    if (m.in_abs) { f[k++] = c0 * m.abs_gain; f[k++] = c1 * m.abs_gain; f[k++] = c2 * m.abs_gain; }
}

// = base_taps() of jc_up.cpp: the 1-D kernel a quarter pixel toward side s (+1 / -1).
inline int base_taps(int base, int s, int* off, float* wt) {
    if (base == 2) {   // Catmull-Rom at t = 0.25
        off[0] = -s; off[1] = 0; off[2] = s; off[3] = 2 * s;
        wt[0] = -0.0703125f; wt[1] = 0.8671875f; wt[2] = 0.2265625f; wt[3] = -0.0234375f;
        return 4;
    }
    off[0] = 0; off[1] = s;
    wt[0] = 0.75f; wt[1] = 0.25f;
    return 2;
}

// = up_base() of jc_up.cpp, with alpha as a fourth channel: the base of the 2x2 block, 16 values
// (4 pixels x RGBA in the output order (0,0), (1,0), (0,1), (1,1)); rows first, then columns.
void base_block(int base, const uint8_t* in, int stride, int W, int H, int px, int py, float* b) {
    int k = 0;
    for (int j = 0; j < 2; j++) {
        int oy[4]; float wy[4];
        const int ny = base_taps(base, j ? 1 : -1, oy, wy);
        for (int i = 0; i < 2; i++) {
            int ox[4]; float wx[4];
            const int nx = base_taps(base, i ? 1 : -1, ox, wx);
            float acc[4] = { 0, 0, 0, 0 };
            for (int ty = 0; ty < ny; ty++) {
                const uint8_t* row = in + (size_t)clampi(py + oy[ty], 0, H - 1) * stride;
                float r[4] = { 0, 0, 0, 0 };
                for (int tx = 0; tx < nx; tx++) {
                    const uint8_t* q = row + (size_t)clampi(px + ox[tx], 0, W - 1) * 4;
                    r[0] += wx[tx] * u8f(q[0]); r[1] += wx[tx] * u8f(q[1]); r[2] += wx[tx] * u8f(q[2]); r[3] += wx[tx] * u8f(q[3]);
                }
                acc[0] += wy[ty] * r[0]; acc[1] += wy[ty] * r[1]; acc[2] += wy[ty] * r[2]; acc[3] += wy[ty] * r[3];
            }
            b[k++] = acc[0]; b[k++] = acc[1]; b[k++] = acc[2]; b[k++] = acc[3];
        }
    }
}

// = mlp_forward_linear() of jc_mlp.h.
void forward(const Model& m, const float* in, float* out) {
    float bufA[kMaxW], bufB[kMaxW];
    const float* p = m.weights;
    const float* cur = in;
    int ncur = m.nin;
    float* nxt = bufA;
    for (int l = 0; l < m.nlayers; l++) {
        const int nh = m.hidden[l];
        const float* W = p; const float* b = W + (size_t)nh * ncur;
        for (int j = 0; j < nh; j++) {
            float s = b[j]; const float* w = W + (size_t)j * ncur;
            for (int i = 0; i < ncur; i++) s += w[i] * cur[i];
            nxt[j] = s > 0 ? s : m.leak * s;
        }
        p = b + nh;
        cur = nxt; ncur = nh;
        nxt = (nxt == bufA) ? bufB : bufA;
    }
    const float* W = p; const float* b = W + (size_t)m.nout * ncur;
    for (int j = 0; j < m.nout; j++) {
        float s = b[j]; const float* w = W + (size_t)j * ncur;
        for (int i = 0; i < ncur; i++) s += w[i] * cur[i];
        out[j] = s;
    }
}

// = to_u8() of jc_up.cpp
inline uint8_t to_u8(float v) {
    const long q = std::lround(v * 255.0f);
    return (uint8_t)(q < 0 ? 0 : q > 255 ? 255 : q);
}

// ---------------------------------------------------------------- threads
// The platform's thread API, so that a failed start is a return code rather than an exception.
struct ThreadStart { void (*fn)(void*); void* arg; };   // what a new thread runs: fn(arg)
#if defined(_WIN32)
struct NativeThread {
    HANDLE h = nullptr;
    static unsigned __stdcall entry(void* p) { const ThreadStart* s = (const ThreadStart*)p; s->fn(s->arg); return 0; }
    bool start(const ThreadStart* s) { h = (HANDLE)_beginthreadex(nullptr, 0, entry, (void*)s, 0, nullptr); return h != nullptr; }   // 0 = could not start
    void join() { if (h) { WaitForSingleObject(h, INFINITE); CloseHandle(h); h = nullptr; } }
};
static int hardware_threads() { SYSTEM_INFO si; GetSystemInfo(&si); return si.dwNumberOfProcessors > 0 ? (int)si.dwNumberOfProcessors : 1; }
#else
struct NativeThread {
    pthread_t t; bool started = false;
    static void* entry(void* p) { const ThreadStart* s = (const ThreadStart*)p; s->fn(s->arg); return nullptr; }
    bool start(const ThreadStart* s) { started = pthread_create(&t, nullptr, entry, (void*)s) == 0; return started; }   // an error code = could not start
    void join() { if (started) { pthread_join(t, nullptr); started = false; } }
};
static int hardware_threads() { const long n = sysconf(_SC_NPROCESSORS_ONLN); return n > 0 ? (int)n : 1; }
#endif

// Runs fn(ctx, row0, row1) over [0, H) on up to `threads` threads, the calling thread one of
// them. Always completes every row: a thread that cannot be created costs nothing but its rows,
// which the calling thread does after its own; every thread that was started is joined before
// returning. The per-thread bookkeeping (at most kMaxThreads entries) lives on the stack.
static const int kMaxThreads = 256;
typedef void (*RowFn)(void* ctx, int r0, int r1);
struct RowJob { RowFn fn; void* ctx; int r0, r1; ThreadStart start; };
static void row_job_entry(void* p) { RowJob* j = (RowJob*)p; j->fn(j->ctx, j->r0, j->r1); }
static void parallel_rows(int H, int threads, RowFn fn, void* ctx) {
    threads = (std::min)(threads, (std::max)(1, H / 8));   // at least 8 input rows per thread: thread creation is not free
    threads = (std::min)(threads, kMaxThreads);
    if (threads <= 1) { fn(ctx, 0, H); return; }
    RowJob jobs[kMaxThreads];
    NativeThread pool[kMaxThreads];
    bool started[kMaxThreads];
    for (int t = 1; t < threads; t++) {
        RowJob& j = jobs[t];
        j.fn = fn; j.ctx = ctx; j.r0 = (int)((int64_t)H * t / threads); j.r1 = (int)((int64_t)H * (t + 1) / threads);
        j.start.fn = row_job_entry; j.start.arg = &j;
        started[t] = pool[t].start(&j.start);
    }
    fn(ctx, 0, (int)((int64_t)H / threads));
    for (int t = 1; t < threads; t++) if (!started[t]) fn(ctx, jobs[t].r0, jobs[t].r1);   // the rows of the threads that could not start
    for (int t = 1; t < threads; t++) if (started[t]) pool[t].join();
}

// ---------------------------------------------------------------- the pass
struct Ctx {
    const Model* m;
    const uint8_t* in; uint8_t* out;
    int W, H, in_stride, out_stride;
    float rscale;
};
// = up_apply() + up_compose() over the input rows [r0, r1): each input pixel's 2x2 output block.
static void up_rows(void* p, int r0, int r1) {
    const Ctx& c = *(const Ctx*)p;
    const Model& m = *c.m;
    float f[kMaxNin], r[kNout], b[16];
    for (int py = r0; py < r1; py++) {
        for (int px = 0; px < c.W; px++) {
            features(m, c.in, c.in_stride, c.W, c.H, px, py, f);
            forward(m, f, r);
            base_block(m.base, c.in, c.in_stride, c.W, c.H, px, py, b);
            int k = 0;
            for (int j = 0; j < 2; j++)
                for (int i = 0; i < 2; i++) {
                    uint8_t* q = c.out + (size_t)(2 * py + j) * c.out_stride + (size_t)(2 * px + i) * 4;
                    const float* bb = b + (size_t)(2 * j + i) * 4;
                    for (int ch = 0; ch < 3; ch++) { q[ch] = to_u8(bb[ch] + c.rscale * r[k]); k++; }
                    q[3] = to_u8(bb[3]);
                }
        }
    }
}

} // namespace

int jpegc_up2x_weights(jpegc_up2x_model model) { return model == JPEGC_UP2X_BEST ? kBest.nweights : model == JPEGC_UP2X_FAST ? kFast.nweights : 0; }

jpegc_up2x_status jpegc_up2x_rgba(const uint8_t* in, int width, int height, int in_stride,
                                  uint8_t* out, int out_stride, jpegc_up2x_model model, int strength, int threads) {
    if (!in || !out || width < 1 || height < 1) return JPEGC_UP2X_BAD_ARGUMENT;
    if ((int64_t)width * 4 > in_stride || (int64_t)width * 8 > out_stride) return JPEGC_UP2X_BAD_ARGUMENT;
    if (width > 0x3FFFFFFF || height > 0x3FFFFFFF) return JPEGC_UP2X_BAD_ARGUMENT;   // 2 * width and 2 * height must fit an int
    if (model != JPEGC_UP2X_BEST && model != JPEGC_UP2X_FAST) return JPEGC_UP2X_BAD_ARGUMENT;
    if (strength < 1 || strength > 300) return JPEGC_UP2X_BAD_ARGUMENT;
    {   // the buffers must not overlap (the output is written while the input is still read)
        const uintptr_t i0 = (uintptr_t)in, i1 = i0 + (uintptr_t)(height - 1) * (uintptr_t)in_stride + (uintptr_t)width * 4;
        const uintptr_t o0 = (uintptr_t)out, o1 = o0 + (uintptr_t)(2 * height - 1) * (uintptr_t)out_stride + (uintptr_t)width * 8;
        if (i0 < o1 && o0 < i1) return JPEGC_UP2X_BAD_ARGUMENT;
    }
    if (threads <= 0) threads = hardware_threads();
    Ctx c;
    c.m = model == JPEGC_UP2X_BEST ? &kBest : &kFast;
    c.in = in; c.out = out; c.W = width; c.H = height; c.in_stride = in_stride; c.out_stride = out_stride;
    c.rscale = (float)strength / 50.0f;
    parallel_rows(height, threads, up_rows, &c);
    return JPEGC_UP2X_OK;
}
