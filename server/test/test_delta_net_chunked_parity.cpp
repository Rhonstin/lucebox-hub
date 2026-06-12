// Parity test: build_delta_net_chunked vs the fused ggml_gated_delta_net
// kernel on identical random inputs (Qwen3.6-27B delta-net shapes).
//
// The chunked port is gated off in prod (DFLASH27B_CHUNKED) with a note that
// it "produces correct shape but slightly wrong final state". This test
// localizes the divergence: per-token output diff and final-state diff are
// reported separately, across n_tokens that exercise the pad path (16, 23),
// the exact-chunk path (64, 128) and the multi-chunk path (100, 256).
//
// Usage: test_delta_net_chunked_parity [cpu|cuda]   (default cuda)

#include "delta_net_chunked.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using namespace dflash::common;

// Qwen3.6-27B delta-net dims (after the q/k head repeat the call site does).
static const int S    = 128;  // head dim (k and v)
static const int H    = 48;   // num_v_heads (q/k repeated 16 -> 48)
static const int SEQS = 1;

struct Inputs {
    std::vector<float> q, k, v, g, b, s;
};

static Inputs make_inputs(int n_tokens, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nrm(0.0f, 1.0f);
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);

    Inputs in;
    in.q.resize((size_t)S * H * n_tokens);
    in.k.resize((size_t)S * H * n_tokens);
    in.v.resize((size_t)S * H * n_tokens);
    in.g.resize((size_t)1 * H * n_tokens);
    in.b.resize((size_t)1 * H * n_tokens);
    in.s.resize((size_t)S * S * H);

    // q/k l2-normalized per row like the graph's ggml_l2_norm.
    auto fill_norm = [&](std::vector<float> & t) {
        for (size_t r = 0; r < t.size() / S; r++) {
            float ss = 0;
            for (int i = 0; i < S; i++) { t[r * S + i] = nrm(rng); ss += t[r * S + i] * t[r * S + i]; }
            const float inv = 1.0f / std::sqrt(ss + 1e-6f);
            for (int i = 0; i < S; i++) t[r * S + i] *= inv;
        }
    };
    fill_norm(in.q);
    fill_norm(in.k);
    for (auto & x : in.v) x = nrm(rng) * 0.5f;
    // g: log-decay, realistic range from -A*softplus: small negative values.
    for (auto & x : in.g) x = -0.05f - 0.6f * uni(rng);
    // beta: sigmoid output.
    for (auto & x : in.b) x = 0.05f + 0.9f * uni(rng);
    for (auto & x : in.s) x = nrm(rng) * 0.05f;
    return in;
}

struct Diff { float max_abs = 0, max_rel = 0; };

static Diff diff(const std::vector<float> & a, const std::vector<float> & b) {
    Diff d;
    for (size_t i = 0; i < a.size(); i++) {
        const float ab = std::fabs(a[i] - b[i]);
        const float rel = ab / (std::fabs(b[i]) + 1e-5f);
        if (ab > d.max_abs) d.max_abs = ab;
        if (rel > d.max_rel) d.max_rel = rel;
    }
    return d;
}

// Run one path; chunked=false -> fused kernel. Returns output [S*H*n] and
// final state [S*S*H].
static bool run_path(ggml_backend_t backend, const Inputs & in, int n_tokens,
                     bool chunked,
                     std::vector<float> & out, std::vector<float> & state) {
    ggml_init_params ip{};
    ip.mem_size   = 512 * 1024 * 1024;
    ip.no_alloc   = true;
    ggml_context * ctx = ggml_init(ip);

    // Layout matches build_delta_net_block at the fused-op call site:
    // q/k/v [S, H, n_tokens, 1], g/b [1, H, n_tokens, 1], s [S, S, H, 1].
    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H, n_tokens, SEQS);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H, n_tokens, SEQS);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H, n_tokens, SEQS);
    ggml_tensor * g = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H, n_tokens, SEQS);
    ggml_tensor * b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H, n_tokens, SEQS);
    ggml_tensor * s = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, S, H, SEQS);
    for (ggml_tensor * t : {q, k, v, g, b, s}) ggml_set_input(t);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8192, false);

    ggml_tensor * t_out = nullptr;
    ggml_tensor * t_state = nullptr;
    if (chunked) {
        auto r = build_delta_net_chunked(ctx, q, k, v, g, b, s);
        t_out   = ggml_cont(ctx, r.output);     // [S, H, n_tokens, 1]
        t_state = ggml_cont(ctx, r.new_state);  // [S, S, H, 1]
    } else {
        ggml_tensor * result = ggml_gated_delta_net(ctx, q, k, v, g, b, s);
        const size_t r_elt = ggml_element_size(result);
        ggml_tensor * o = ggml_view_4d(ctx, result,
            S, H, n_tokens, SEQS,
            S * r_elt, (size_t)S * H * r_elt, (size_t)S * H * n_tokens * r_elt, 0);
        ggml_tensor * ns = ggml_view_4d(ctx, result,
            S, S, H, SEQS,
            S * r_elt, (size_t)S * S * r_elt, (size_t)S * S * H * r_elt,
            (size_t)S * H * n_tokens * SEQS * r_elt);
        t_out   = ggml_cont(ctx, o);
        t_state = ggml_cont(ctx, ns);
    }
    ggml_set_output(t_out);
    ggml_set_output(t_state);
    ggml_build_forward_expand(gf, t_out);
    ggml_build_forward_expand(gf, t_state);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(alloc, gf)) {
        std::fprintf(stderr, "alloc_graph failed (chunked=%d n=%d)\n", chunked, n_tokens);
        return false;
    }

    ggml_backend_tensor_set(q, in.q.data(), 0, sizeof(float) * in.q.size());
    ggml_backend_tensor_set(k, in.k.data(), 0, sizeof(float) * in.k.size());
    ggml_backend_tensor_set(v, in.v.data(), 0, sizeof(float) * in.v.size());
    ggml_backend_tensor_set(g, in.g.data(), 0, sizeof(float) * in.g.size());
    ggml_backend_tensor_set(b, in.b.data(), 0, sizeof(float) * in.b.size());
    ggml_backend_tensor_set(s, in.s.data(), 0, sizeof(float) * in.s.size());

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "compute failed (chunked=%d n=%d)\n", chunked, n_tokens);
        return false;
    }

    out.resize((size_t)S * H * n_tokens);
    state.resize((size_t)S * S * H);
    ggml_backend_tensor_get(t_out, out.data(), 0, sizeof(float) * out.size());
    ggml_backend_tensor_get(t_state, state.data(), 0, sizeof(float) * state.size());

    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    return true;
}

// Variant C/D: replicate the REAL integration pattern from
// build_delta_net_block — q/k/v as strided views into a conv_out-like
// buffer, s as a reshape view over a persistent (non-gallocr) backend
// buffer, the final state persisted via ggml_cpy(new_state, s) exactly like
// the after_delta_net label does, and `steps` sequential graph computes
// chained through that persistent state. Sequential reference uses the
// fused kernel with the same persistent-state pattern.
static bool run_real_pattern(ggml_backend_t backend, int n_tokens, int steps,
                             bool chunked, float g_lo, float g_hi,
                             std::vector<float> & out_last,
                             std::vector<float> & state_after) {
    const int conv_channels = 2 * S * 16 + S * H;  // q,k (16 heads) + v rows live here
    (void)conv_channels;

    // Persistent state buffer (outside gallocr), like cache_.ssm_state.
    ggml_init_params pip{};
    pip.mem_size = 16 * 1024;
    pip.no_alloc = true;
    ggml_context * pctx = ggml_init(pip);
    ggml_tensor * ssm_state = ggml_new_tensor_3d(pctx, GGML_TYPE_F32, S, S, H);
    ggml_backend_buffer_t pbuf = ggml_backend_alloc_ctx_tensors(pctx, backend);
    if (!pbuf) return false;

    {
        Inputs in0 = make_inputs(n_tokens, 7);
        ggml_backend_tensor_set(ssm_state, in0.s.data(), 0, sizeof(float) * in0.s.size());
    }

    bool ok = true;
    for (int step = 0; step < steps && ok; step++) {
        Inputs in = make_inputs(n_tokens, 1000 + step);
        // Override g with the requested range.
        {
            std::mt19937 rng(2000 + step);
            std::uniform_real_distribution<float> uni(0.0f, 1.0f);
            for (auto & x : in.g) x = g_lo + (g_hi - g_lo) * uni(rng);
        }

        ggml_init_params ip{};
        ip.mem_size = 512 * 1024 * 1024;
        ip.no_alloc = true;
        ggml_context * ctx = ggml_init(ip);

        // Strided storage: row = [q(S) | k(S) | v(S) | junk(S)] per (h, t).
        const int row = 4 * S;
        ggml_tensor * big = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, row, H, n_tokens);
        ggml_set_input(big);
        const size_t elt = sizeof(float);
        auto view_qkv = [&](int64_t off) {
            return ggml_view_4d(ctx, big, S, H, n_tokens, SEQS,
                                row * elt, row * H * elt, row * H * n_tokens * elt,
                                off * elt);
        };
        ggml_tensor * q = view_qkv(0);
        ggml_tensor * k = view_qkv(S);
        ggml_tensor * v = view_qkv(2 * S);
        ggml_tensor * g = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H, n_tokens, SEQS);
        ggml_tensor * b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H, n_tokens, SEQS);
        ggml_set_input(g);
        ggml_set_input(b);

        ggml_tensor * s = ggml_reshape_4d(ctx, ssm_state, S, S, H, SEQS);

        ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8192, false);

        ggml_tensor * t_out = nullptr;
        if (chunked) {
            auto r = build_delta_net_chunked(ctx, q, k, v, g, b, s);
            t_out = ggml_cont(ctx, r.output);
            ggml_set_output(t_out);
            ggml_build_forward_expand(gf, t_out);
            ggml_build_forward_expand(gf, ggml_cpy(ctx, r.new_state, s));
        } else {
            ggml_tensor * result = ggml_gated_delta_net(ctx, q, k, v, g, b, s);
            const size_t r_elt = ggml_element_size(result);
            ggml_tensor * o = ggml_view_4d(ctx, result,
                S, H, n_tokens, SEQS,
                S * r_elt, (size_t)S * H * r_elt, (size_t)S * H * n_tokens * r_elt, 0);
            ggml_tensor * ns = ggml_view_4d(ctx, result,
                S, S, H, SEQS,
                S * r_elt, (size_t)S * S * r_elt, (size_t)S * S * H * r_elt,
                (size_t)S * H * n_tokens * SEQS * r_elt);
            t_out = ggml_cont(ctx, o);
            ggml_set_output(t_out);
            ggml_build_forward_expand(gf, t_out);
            ggml_build_forward_expand(gf, ggml_cpy(ctx, ns, s));
        }

        ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!ggml_gallocr_alloc_graph(alloc, gf)) { ok = false; ggml_free(ctx); break; }

        // Fill the strided storage: junk everywhere, then q/k/v into views.
        std::vector<float> big_host((size_t)row * H * n_tokens, 1234.5f);
        for (int t = 0; t < n_tokens; t++) {
            for (int h = 0; h < H; h++) {
                float * dst = big_host.data() + ((size_t)t * H + h) * row;
                std::memcpy(dst,         in.q.data() + ((size_t)t * H + h) * S, S * elt);
                std::memcpy(dst + S,     in.k.data() + ((size_t)t * H + h) * S, S * elt);
                std::memcpy(dst + 2 * S, in.v.data() + ((size_t)t * H + h) * S, S * elt);
            }
        }
        ggml_backend_tensor_set(big, big_host.data(), 0, sizeof(float) * big_host.size());
        ggml_backend_tensor_set(g, in.g.data(), 0, sizeof(float) * in.g.size());
        ggml_backend_tensor_set(b, in.b.data(), 0, sizeof(float) * in.b.size());

        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) { ok = false; }
        if (ok && step == steps - 1) {
            out_last.resize((size_t)S * H * n_tokens);
            ggml_backend_tensor_get(t_out, out_last.data(), 0, sizeof(float) * out_last.size());
        }
        ggml_gallocr_free(alloc);
        ggml_free(ctx);
    }

    if (ok) {
        state_after.resize((size_t)S * S * H);
        ggml_backend_tensor_get(ssm_state, state_after.data(), 0,
                                sizeof(float) * state_after.size());
    }
    ggml_backend_buffer_free(pbuf);
    ggml_free(pctx);
    return ok;
}

int main(int argc, char ** argv) {
    const bool use_cpu = (argc > 1 && std::strcmp(argv[1], "cpu") == 0);
    ggml_backend_t backend = nullptr;
#ifdef GGML_USE_CUDA
    if (!use_cpu) backend = ggml_backend_cuda_init(0);
#endif
    if (!backend) backend = ggml_backend_cpu_init();
    std::printf("backend: %s\n", ggml_backend_name(backend));

    const int sizes[] = {16, 23, 64, 100, 128, 256};
    int failures = 0;
    for (int n : sizes) {
        Inputs in = make_inputs(n, 42 + n);
        std::vector<float> out_f, st_f, out_c, st_c;
        if (!run_path(backend, in, n, false, out_f, st_f)) { failures++; continue; }
        if (!run_path(backend, in, n, true,  out_c, st_c)) { failures++; continue; }

        // Per-token max diff to localize where the output starts diverging.
        float first_bad_tok = -1;
        for (int t = 0; t < n && first_bad_tok < 0; t++) {
            for (int i = 0; i < S * H; i++) {
                if (std::fabs(out_f[(size_t)t * S * H + i] - out_c[(size_t)t * S * H + i]) > 1e-3f) {
                    first_bad_tok = (float)t;
                    break;
                }
            }
        }
        Diff od = diff(out_c, out_f);
        Diff sd = diff(st_c, st_f);
        const bool ok = od.max_abs < 1e-3f && sd.max_abs < 1e-3f;
        if (!ok) failures++;
        std::printf("n=%3d  out max_abs=%.6f max_rel=%.4f  state max_abs=%.6f max_rel=%.4f  "
                    "first_bad_tok=%d  %s\n",
                    n, od.max_abs, od.max_rel, sd.max_abs, sd.max_rel,
                    (int)first_bad_tok, ok ? "OK" : "FAIL");
    }

    // Real-integration variants: strided views + persistent-state cpy +
    // chained steps + g ranges (mild like the synthetic test, and strongly
    // negative like fast-forgetting heads).
    struct Variant { const char * name; int n; int steps; float g_lo, g_hi; };
    const Variant variants[] = {
        {"views+cpy n=16 1step mild-g",   16, 1, -0.65f, -0.05f},
        {"views+cpy n=16 8step mild-g",   16, 8, -0.65f, -0.05f},
        {"views+cpy n=16 8step strong-g", 16, 8, -12.0f, -0.5f},
        {"views+cpy n=256 2step strong-g", 256, 2, -12.0f, -0.5f},
    };
    for (const auto & vt : variants) {
        std::vector<float> out_f, st_f, out_c, st_c;
        if (!run_real_pattern(backend, vt.n, vt.steps, false, vt.g_lo, vt.g_hi, out_f, st_f) ||
            !run_real_pattern(backend, vt.n, vt.steps, true,  vt.g_lo, vt.g_hi, out_c, st_c)) {
            std::printf("%-34s ERROR\n", vt.name);
            failures++;
            continue;
        }
        Diff od = diff(out_c, out_f);
        Diff sd = diff(st_c, st_f);
        const bool ok = od.max_abs < 1e-3f && sd.max_abs < 1e-3f;
        if (!ok) failures++;
        std::printf("%-34s out max_abs=%.6f  state max_abs=%.6f  %s\n",
                    vt.name, od.max_abs, sd.max_abs, ok ? "OK" : "FAIL");
    }

    ggml_backend_free(backend);
    std::printf(failures ? "PARITY FAIL (%d)\n" : "PARITY OK\n", failures);
    return failures ? 1 : 0;
}
