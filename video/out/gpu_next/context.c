/*
 * This file is part of mpv.
 *
 * mpv is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * mpv is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with mpv.  If not, see <https://www.gnu.org/licenses/>.
 */

#include <libplacebo/config.h>

#ifdef PL_HAVE_D3D11
#include <libplacebo/d3d11.h>
#endif

#ifdef PL_HAVE_OPENGL
#include <libplacebo/opengl.h>
#endif

#include "context.h"
#include "config.h"
#include "common/common.h"
#include "video/out/libmpv.h"
#include "video/out/placebo/utils.h"
#include "video/out/gpu_next/libmpv_gpu_next.h"

#if HAVE_D3D11
#include "mpv/render_d3d11.h"
#include "osdep/windows_utils.h"
#include "video/out/d3d11/ra_d3d11.h"
#include "video/out/d3d11/context.h"
#endif

#if HAVE_GL
#include "mpv/render_gl.h"
#include "video/out/opengl/context.h"
#include "video/out/opengl/ra_gl.h"
# if HAVE_EGL
#include <EGL/egl.h>
# endif
#endif

#if HAVE_VULKAN
#include "video/out/vulkan/context.h"
#endif

#if HAVE_D3D11
static bool d3d11_pl_init(struct vo *vo, struct gpu_ctx *ctx,
                          struct ra_ctx_opts *ctx_opts)
{
#if !defined(PL_HAVE_D3D11)
    MP_MSG(ctx, vo->probing ? MSGL_V : MSGL_ERR,
           "libplacebo was built without D3D11 support.\n");
    return false;
#else // defined(PL_HAVE_D3D11)
    bool success = false;

    ID3D11Device   *device    = ra_d3d11_get_device(ctx->ra_ctx->ra);
    IDXGISwapChain *swapchain = ra_d3d11_ctx_get_swapchain(ctx->ra_ctx);
    if (!device || !swapchain) {
        mp_err(ctx->log,
               "Failed to receive required components from the mpv d3d11 "
               "context! (device: %s, swap chain: %s)\n",
               device    ? "OK" : "failed",
               swapchain ? "OK" : "failed");
        goto err_out;
    }

    pl_d3d11 d3d11 = pl_d3d11_create(ctx->pllog,
        pl_d3d11_params(
            .device = device,
        )
    );
    if (!d3d11) {
        mp_err(ctx->log, "Failed to acquire a d3d11 libplacebo context!\n");
        goto err_out;
    }
    ctx->gpu = d3d11->gpu;

    mppl_log_set_probing(ctx->pllog, false);

    struct pl_d3d11_swapchain_params *params = pl_d3d11_swapchain_params(
         .swapchain = swapchain,
    );
    ra_d3d11_ctx_set_swapchain_params(ctx->ra_ctx, params);
    ctx->swapchain = pl_d3d11_create_swapchain(d3d11, params);
    if (!ctx->swapchain) {
        mp_err(ctx->log, "Failed to acquire a d3d11 libplacebo swap chain!\n");
        goto err_out;
    }

    success = true;

err_out:
    SAFE_RELEASE(swapchain);
    SAFE_RELEASE(device);

    return success;
#endif // defined(PL_HAVE_D3D11)
}
#endif // HAVE_D3D11

struct gpu_ctx *gpu_ctx_create(struct vo *vo, struct ra_ctx_opts *ctx_opts)
{
    struct gpu_ctx *ctx = talloc_zero(NULL, struct gpu_ctx);
    ctx->log = vo->log;
    ctx->ra_ctx = ra_ctx_create(vo, *ctx_opts);
    if (!ctx->ra_ctx)
        goto err_out;

#if HAVE_VULKAN
    struct mpvk_ctx *vkctx = ra_vk_ctx_get(ctx->ra_ctx);
    if (vkctx) {
        ctx->pllog = vkctx->pllog;
        ctx->gpu = vkctx->gpu;
        ctx->swapchain = vkctx->swapchain;
        return ctx;
    }
#endif

    ctx->pllog = mppl_log_create(ctx, ctx->log);
    if (!ctx->pllog)
        goto err_out;

    mppl_log_set_probing(ctx->pllog, vo->probing);

#if HAVE_D3D11
    if (ra_is_d3d11(ctx->ra_ctx->ra)) {
        if (!d3d11_pl_init(vo, ctx, ctx_opts))
            goto err_out;

        return ctx;
    }
#endif

#if HAVE_GL && defined(PL_HAVE_OPENGL)
    if (ra_is_gl(ctx->ra_ctx->ra)) {
        struct GL *gl = ra_gl_get(ctx->ra_ctx->ra);
        struct pl_opengl_params params = *pl_opengl_params(
            .debug = ctx_opts->debug,
            .allow_software = ctx_opts->allow_sw,
            .get_proc_addr_ex = (void *) gl->get_fn,
            .proc_ctx = gl->fn_ctx,
        );
# if HAVE_EGL
        params.egl_display = eglGetCurrentDisplay();
        params.egl_context = eglGetCurrentContext();
# endif
        pl_opengl opengl = pl_opengl_create(ctx->pllog, &params);
        if (!opengl)
            goto err_out;
        ctx->gpu = opengl->gpu;

        mppl_log_set_probing(ctx->pllog, false);

        ctx->swapchain = pl_opengl_create_swapchain(opengl, pl_opengl_swapchain_params(
            .max_swapchain_depth = vo->opts->swapchain_depth,
            .framebuffer.flipped = gl->flipped,
        ));
        if (!ctx->swapchain)
            goto err_out;

        return ctx;
    }
#elif HAVE_GL
    if (ra_is_gl(ctx->ra_ctx->ra)) {
        MP_MSG(ctx, vo->probing ? MSGL_V : MSGL_ERR,
            "libplacebo was built without OpenGL support.\n");
    }
#endif

err_out:
    gpu_ctx_destroy(&ctx);
    return NULL;
}

bool gpu_ctx_resize(struct gpu_ctx *ctx, int w, int h)
{
#if HAVE_VULKAN
    if (ra_vk_ctx_get(ctx->ra_ctx))
        // vulkan RA handles this by itself
        return true;
#endif

    return pl_swapchain_resize(ctx->swapchain, &w, &h);
}

void gpu_ctx_destroy(struct gpu_ctx **ctxp)
{
    struct gpu_ctx *ctx = *ctxp;
    if (!ctx)
        return;
    if (!ctx->ra_ctx)
        goto skip_common_pl_cleanup;

#if HAVE_VULKAN
    if (ra_vk_ctx_get(ctx->ra_ctx))
        // vulkan RA context handles pl cleanup by itself,
        // skip common local clean-up.
        goto skip_common_pl_cleanup;
#endif

    if (ctx->swapchain)
        pl_swapchain_destroy(&ctx->swapchain);

    if (ctx->gpu) {
#if HAVE_GL && defined(PL_HAVE_OPENGL)
        if (ra_is_gl(ctx->ra_ctx->ra)) {
            pl_opengl opengl = pl_opengl_get(ctx->gpu);
            pl_opengl_destroy(&opengl);
        }
#endif

#if HAVE_D3D11 && defined(PL_HAVE_D3D11)
        if (ra_is_d3d11(ctx->ra_ctx->ra)) {
            pl_d3d11 d3d11 = pl_d3d11_get(ctx->gpu);
            pl_d3d11_destroy(&d3d11);
        }
#endif
    }

    if (ctx->pllog)
        pl_log_destroy(&ctx->pllog);

skip_common_pl_cleanup:
    ra_ctx_destroy(&ctx->ra_ctx);

    talloc_free(ctx);
    *ctxp = NULL;
}

#if HAVE_GL && defined(PL_HAVE_OPENGL)

struct priv_gl {
    pl_log pl_log;
    pl_opengl gl;
    pl_tex target;
    int target_fbo;
    int target_w, target_h;
};

static bool pl_callback_gl(void *priv)
{
    mpv_opengl_init_params *gl_params = priv;
    return gl_params->get_proc_address(gl_params->get_proc_address_ctx,
                                       "glGetString") != NULL;
}

static int gl_init(struct libmpv_gpu_next_context *ctx, mpv_render_param *params)
{
    mpv_opengl_init_params *gl_params =
        get_mpv_render_param(params, MPV_RENDER_PARAM_OPENGL_INIT_PARAMS, NULL);
    if (!gl_params || !gl_params->get_proc_address)
        return MPV_ERROR_INVALID_PARAMETER;

    struct priv_gl *p = talloc_zero(ctx, struct priv_gl);
    // destroy() runs even after a failed init(), so publish priv early.
    ctx->priv = p;

    p->pl_log = mppl_log_create(ctx, ctx->log);
    if (!p->pl_log)
        return MPV_ERROR_VO_INIT_FAILED;
    mppl_log_set_probing(p->pl_log, false);

    p->gl = pl_opengl_create(p->pl_log, pl_opengl_params(
        .get_proc_addr_ex = (pl_voidfunc_t (*)(void *, const char *))
                            gl_params->get_proc_address,
        .proc_ctx = gl_params->get_proc_address_ctx,
        .make_current = pl_callback_gl,
        .priv = gl_params,
    ));
    if (!p->gl)
        return MPV_ERROR_UNSUPPORTED;

    ctx->pl_log = p->pl_log;
    ctx->gpu = p->gl->gpu;
    ctx->rr = pl_renderer_create(ctx->pl_log, ctx->gpu);
    return ctx->rr ? 0 : MPV_ERROR_VO_INIT_FAILED;
}

static int gl_wrap_fbo(struct libmpv_gpu_next_context *ctx,
                       mpv_render_param *params, pl_tex *out_tex)
{
    struct priv_gl *p = ctx->priv;

    mpv_opengl_fbo *fbo =
        get_mpv_render_param(params, MPV_RENDER_PARAM_OPENGL_FBO, NULL);
    if (!fbo || fbo->w <= 0 || fbo->h <= 0)
        return MPV_ERROR_INVALID_PARAMETER;

    if (!p->target || p->target_fbo != fbo->fbo ||
        p->target_w != fbo->w || p->target_h != fbo->h)
    {
        pl_tex_destroy(ctx->gpu, &p->target);
        p->target = pl_opengl_wrap(ctx->gpu, pl_opengl_wrap_params(
            .framebuffer = fbo->fbo,
            .width = fbo->w,
            .height = fbo->h,
            .iformat = fbo->internal_format,
        ));
        if (!p->target) {
            MP_ERR(ctx, "Failed to wrap the provided FBO as a texture.\n");
            return MPV_ERROR_GENERIC;
        }
        p->target_fbo = fbo->fbo;
        p->target_w = fbo->w;
        p->target_h = fbo->h;
    }
    *out_tex = p->target;
    return 0;
}

static void gl_destroy(struct libmpv_gpu_next_context *ctx)
{
    struct priv_gl *p = ctx->priv;
    if (!p)
        return;

    if (ctx->gpu)
        pl_tex_destroy(ctx->gpu, &p->target);
    pl_renderer_destroy(&ctx->rr);
    pl_opengl_destroy(&p->gl);
    pl_log_destroy(&p->pl_log);
}

const struct libmpv_gpu_next_context_fns libmpv_gpu_next_context_gl = {
    .api_name = MPV_RENDER_API_TYPE_OPENGL,
    .init = gl_init,
    .wrap_fbo = gl_wrap_fbo,
    .destroy = gl_destroy,
};

#endif // HAVE_GL && defined(PL_HAVE_OPENGL)

#if HAVE_D3D11 && defined(PL_HAVE_D3D11)

struct priv_d3d11 {
    pl_log pl_log;
    pl_d3d11 d3d11;
    pl_tex target;
    ID3D11Texture2D *target_tex;
    int target_w, target_h;
};

// Only used to map hwdec frames; has no ra_ctx_fns and must not be passed to
// ra_ctx_destroy().
static void d3d11_create_ra(struct libmpv_gpu_next_context *ctx, ID3D11Device *dev)
{
    struct ra_ctx *ra_ctx = talloc_zero(ctx, struct ra_ctx);
    ra_ctx->log = ctx->log;
    ra_ctx->global = ctx->global;
    if (spirv_compiler_init(ra_ctx) &&
        (ra_ctx->ra = ra_d3d11_create(dev, ctx->log, ra_ctx->spirv)))
    {
        ctx->ra_ctx = ra_ctx;
        return;
    }
    MP_ERR(ctx, "Failed to create the D3D11 hwdec interop context.\n");
    if (ra_ctx->spirv && ra_ctx->spirv->fns->uninit)
        ra_ctx->spirv->fns->uninit(ra_ctx);
    talloc_free(ra_ctx);
}

static void d3d11_destroy_ra(struct ra_ctx *ra_ctx)
{
    if (!ra_ctx)
        return;

    if (ra_ctx->ra) {
        ra_ctx->ra->fns->destroy(ra_ctx->ra);
        ra_ctx->ra = NULL;
    }
    if (ra_ctx->spirv && ra_ctx->spirv->fns->uninit)
        ra_ctx->spirv->fns->uninit(ra_ctx);
    talloc_free(ra_ctx);
}

static int d3d11_init(struct libmpv_gpu_next_context *ctx, mpv_render_param *params)
{
    struct priv_d3d11 *p = talloc_zero(ctx, struct priv_d3d11);
    // destroy() runs even after a failed init(), so publish priv early.
    ctx->priv = p;

    mpv_d3d11_init_params *init =
        get_mpv_render_param(params, MPV_RENDER_PARAM_D3D11_INIT_PARAMS, NULL);
    if (!init || !init->device)
        return MPV_ERROR_INVALID_PARAMETER;

    p->pl_log = mppl_log_create(ctx, ctx->log);
    if (!p->pl_log)
        return MPV_ERROR_VO_INIT_FAILED;
    mppl_log_set_probing(p->pl_log, false);

    p->d3d11 = pl_d3d11_create(p->pl_log, pl_d3d11_params(.device = init->device));
    if (!p->d3d11)
        return MPV_ERROR_UNSUPPORTED;

    ctx->pl_log = p->pl_log;
    ctx->gpu = p->d3d11->gpu;
    ctx->rr = pl_renderer_create(ctx->pl_log, ctx->gpu);
    if (!ctx->rr)
        return MPV_ERROR_VO_INIT_FAILED;

    d3d11_create_ra(ctx, init->device);
    return 0;
}

static int d3d11_wrap_fbo(struct libmpv_gpu_next_context *ctx,
                          mpv_render_param *params, pl_tex *out_tex)
{
    struct priv_d3d11 *p = ctx->priv;

    mpv_d3d11_fbo *fbo =
        get_mpv_render_param(params, MPV_RENDER_PARAM_D3D11_FBO, NULL);
    if (!fbo || !fbo->tex)
        return MPV_ERROR_INVALID_PARAMETER;

    // mpv_d3d11_fbo carries no dimensions, so query them from the texture.
    ID3D11Texture2D *tex = (ID3D11Texture2D *)fbo->tex;
    D3D11_TEXTURE2D_DESC desc = {0};
    ID3D11Texture2D_GetDesc(tex, &desc);
    if (!desc.Width || !desc.Height)
        return MPV_ERROR_INVALID_PARAMETER;

    if (!p->target || p->target_tex != tex ||
        p->target_w != (int)desc.Width || p->target_h != (int)desc.Height)
    {
        pl_tex_destroy(ctx->gpu, &p->target);
        p->target = pl_d3d11_wrap(ctx->gpu, pl_d3d11_wrap_params(
            .tex = (ID3D11Resource *)tex,
            .w = desc.Width,
            .h = desc.Height,
        ));
        if (!p->target) {
            MP_ERR(ctx, "Failed to wrap the provided D3D11 texture.\n");
            return MPV_ERROR_GENERIC;
        }
        p->target_tex = tex;
        p->target_w = desc.Width;
        p->target_h = desc.Height;
    }
    *out_tex = p->target;
    return 0;
}

static void d3d11_done_frame(struct libmpv_gpu_next_context *ctx)
{
    pl_gpu_flush(ctx->gpu);
}

static void d3d11_destroy(struct libmpv_gpu_next_context *ctx)
{
    struct priv_d3d11 *p = ctx->priv;
    if (!p)
        return;

    if (ctx->gpu) {
        pl_gpu_flush(ctx->gpu);
        pl_tex_destroy(ctx->gpu, &p->target);
    }
    pl_renderer_destroy(&ctx->rr);
    d3d11_destroy_ra(ctx->ra_ctx);
    ctx->ra_ctx = NULL;
    pl_d3d11_destroy(&p->d3d11);
    pl_log_destroy(&p->pl_log);
}

const struct libmpv_gpu_next_context_fns libmpv_gpu_next_context_d3d11 = {
    .api_name = MPV_RENDER_API_TYPE_D3D11,
    .init = d3d11_init,
    .wrap_fbo = d3d11_wrap_fbo,
    .done_frame = d3d11_done_frame,
    .destroy = d3d11_destroy,
};

#endif // HAVE_D3D11 && defined(PL_HAVE_D3D11)
