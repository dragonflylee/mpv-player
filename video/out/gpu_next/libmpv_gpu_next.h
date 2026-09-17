#pragma once

#include <libplacebo/gpu.h>
#include <libplacebo/renderer.h>

#include "mpv/render.h"

struct ra_ctx;

// Backend-specific glue for the libmpv 'gpu-next' render backend. It creates
// the libplacebo GPU for the client's API and wraps the target surface.
struct libmpv_gpu_next_context {
    struct mp_log *log;
    struct mpv_global *global;
    const struct libmpv_gpu_next_context_fns *fns;

    void *priv;

    // Set by init(); the GPU is owned by the backend, the renderer by us.
    pl_log pl_log;
    pl_gpu gpu;
    pl_renderer rr;

    // Anchor for the hwdec interop; not used for output.
    struct ra_ctx *ra_ctx;
};

struct libmpv_gpu_next_context_fns {
    // The libmpv API type name, see MPV_RENDER_PARAM_API_TYPE.
    const char *api_name;

    // Fill ctx->pl_log/gpu/rr and ctx->priv.
    int (*init)(struct libmpv_gpu_next_context *ctx, mpv_render_param *params);

    // Wrap the surface passed to mpv_render_context_render() into a pl_tex.
    // The result is owned by the backend and reused across frames.
    int (*wrap_fbo)(struct libmpv_gpu_next_context *ctx,
                    mpv_render_param *params, pl_tex *out);

    // Called after rendering a frame. Optional.
    void (*done_frame)(struct libmpv_gpu_next_context *ctx);

    // Free all data in ctx->priv, including ctx->rr and ctx->gpu.
    void (*destroy)(struct libmpv_gpu_next_context *ctx);
};

extern const struct libmpv_gpu_next_context_fns libmpv_gpu_next_context_gl;
extern const struct libmpv_gpu_next_context_fns libmpv_gpu_next_context_d3d11;

