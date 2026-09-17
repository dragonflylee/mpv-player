#pragma once

#include <libplacebo/gpu.h>
#include <libplacebo/log.h>
#include <libplacebo/renderer.h>
#include "mpv/render.h"

// Backend-specific interaction between libmpv and a libplacebo GPU backend
// (initialization and passing FBOs), for the 'gpu-next' render backend.
struct libmpv_gpu_next_context {
    struct mpv_global *global;
    struct mp_log *log;
    void *priv;

    const struct libmpv_gpu_next_context_fns *fns;

    // Set by init(). Owned by the backend implementation.
    pl_log pl_log;
    pl_gpu gpu;
    pl_renderer renderer;

    // Set by init() if the backend provides one. Used only as the anchor for
    // the hwdec interop (ra_ctx->ra), it is not used for output.
    struct ra_ctx *ra_ctx;
};

struct libmpv_gpu_next_context_fns {
    // The libmpv API type name, see MPV_RENDER_PARAM_API_TYPE.
    const char *api_name;
    // Successful init must set ctx->pl_log, ctx->gpu and ctx->renderer.
    int (*init)(struct libmpv_gpu_next_context *ctx, mpv_render_param *params);
    // Wrap the surface passed to mpv_render_context_render() (via the params
    // array) into a pl_tex and return it. The returned object is owned by the
    // backend and is valid until another wrap_fbo() or done_frame() is called.
    int (*wrap_fbo)(struct libmpv_gpu_next_context *ctx,
                    mpv_render_param *params, pl_tex *out);
    // Signal that the pl_tex obtained with wrap_fbo is no longer used.
    void (*done_frame)(struct libmpv_gpu_next_context *ctx);
    // Free all data in ctx->priv.
    void (*destroy)(struct libmpv_gpu_next_context *ctx);
};

extern const struct libmpv_gpu_next_context_fns libmpv_gpu_next_context_gl;
extern const struct libmpv_gpu_next_context_fns libmpv_gpu_next_context_d3d11;

