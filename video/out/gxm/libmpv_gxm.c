#include "common/msg.h"
#include "video/out/gpu/context.h"
#include "video/out/gpu/libmpv_gpu.h"
#include "video/out/gxm/ra_gxm.h"
#include "libmpv/render_gxm.h"

struct priv {
    SceGxmContext *context;
    SceGxmShaderPatcher *shader_patcher;
    struct ra_ctx *ra_ctx;
    struct ra_tex *cur_fbo;
};

struct gxm_fbo_priv {
    SceGxmRenderTarget *render_target;
    SceGxmColorSurface *color_surface;
    SceGxmDepthStencilSurface *depth_stencil_surface;
};

static int init(struct libmpv_gpu_context *ctx, mpv_render_param *params)
{
    MP_VERBOSE(ctx, "Creating libmpv gxm context\n");

    struct priv *p = ctx->priv = talloc_zero(NULL, struct priv);

    mpv_gxm_init_params *init_params =
            get_mpv_render_param(params, MPV_RENDER_PARAM_GXM_INIT_PARAMS, NULL);
    if (!init_params)
        return MPV_ERROR_INVALID_PARAMETER;

    p->context = init_params->context;
    p->shader_patcher = init_params->shader_patcher;
#ifndef HAVE_VITASHARK
    init_params->buffer_index = 0;
#endif

    // initialize a blank ra_ctx to reuse ra_gl_ctx
    p->ra_ctx = talloc_zero(p, struct ra_ctx);
    p->ra_ctx->log = ctx->log;
    p->ra_ctx->global = ctx->global;

    p->ra_ctx->ra = ra_gxm_create(ctx->log, p->context, p->shader_patcher, init_params->buffer_index, init_params->msaa);
    if (!p->ra_ctx->ra)
        return MPV_ERROR_UNSUPPORTED;

    p->cur_fbo = talloc_zero(p, struct ra_tex);
    p->cur_fbo->priv = talloc_zero(p, struct gxm_fbo_priv);

    ctx->ra_ctx = p->ra_ctx;
    return 0;
}

static int wrap_fbo(struct libmpv_gpu_context *ctx, mpv_render_param *params, struct ra_tex **out)
{
    struct priv *p = ctx->priv;
    struct ra *ra = p->ra_ctx->ra;
    struct gxm_fbo_priv *fbo_priv = p->cur_fbo->priv;
    struct ra_format *fmt = NULL;

    mpv_gxm_fbo *fbo = get_mpv_render_param(params, MPV_RENDER_PARAM_GXM_FBO, NULL);
    if (!fbo)
        return MPV_ERROR_INVALID_PARAMETER;

    for (int n = 0; n < ra->num_formats; n++) {
        const struct gxm_format *gf = ra->formats[n]->priv;
        if (gf->format == fbo->format) {
            fmt = ra->formats[n];
            break;
        }
    }

    p->cur_fbo->params = (struct ra_tex_params) {
            .dimensions = 2,
            .w          = fbo->w,
            .h          = fbo->h,
            .d          = 1,
            .format     = fmt,
            .render_dst = true,
            // TODO: blit_src and blit_dst
    };
    fbo_priv->color_surface = fbo->color_surface;
    fbo_priv->render_target = fbo->render_target;
    fbo_priv->depth_stencil_surface = fbo->depth_stencil_surface;

    *out = p->cur_fbo;
    return 0;
}

static void __attribute__((__optimize__("no-optimize-sibling-calls"))) start_frame(struct libmpv_gpu_context *ctx)
{
    struct priv *p = ctx->priv;
    struct gxm_fbo_priv *fbo = p->cur_fbo->priv;
    if (!fbo->render_target || !fbo->color_surface || !fbo->depth_stencil_surface)
        return;
    sceGxmBeginScene(p->context,
                    0,
                    fbo->render_target,
                    NULL,
                    NULL,
                    NULL,
                    fbo->color_surface,
                    fbo->depth_stencil_surface);
}

static void __attribute__((__optimize__("no-optimize-sibling-calls"))) done_frame(struct libmpv_gpu_context *ctx, bool ds)
{
    struct priv *p = ctx->priv;
    struct gxm_fbo_priv *fbo = p->cur_fbo->priv;
    if (!fbo->render_target || !fbo->color_surface || !fbo->depth_stencil_surface)
        return;
    sceGxmEndScene(p->context, NULL, NULL);
}

static void destroy(struct libmpv_gpu_context *ctx)
{
    MP_VERBOSE(ctx, "Destroying libmpv gxm context\n");

    struct ra_ctx *ra_ctx = ctx->ra_ctx;
    if (ra_ctx->ra)
        ra_ctx->ra->fns->destroy(ra_ctx->ra);
}

const struct libmpv_gpu_context_fns libmpv_gpu_context_gxm = {
        .api_name    = MPV_RENDER_API_TYPE_GXM,
        .init        = init,
        .wrap_fbo    = wrap_fbo,
        .done_frame  = done_frame,
        .destroy     = destroy,
        .start_frame = start_frame,
};