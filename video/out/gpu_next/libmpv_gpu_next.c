#include "config.h"
#include "common/msg.h"
#include "libmpv_gpu_next.h"
#include "mpv/client.h"
#include "mpv/render.h"
#include "options/m_config.h"
#include "ta/ta_talloc.h"
#include "video/hwdec.h"
#include "video/out/gpu/hwdec.h"
#include "video/out/gpu/video.h"
#include "video/out/libmpv.h"
#include "video/out/vo.h"
#include "video.h"

struct priv {
    struct libmpv_gpu_next_context *context; // Manages the API (e.g., GL, D3D11)
    struct pl_video *video_engine;           // Manages synchronous libplacebo rendering
    struct m_config_cache *gl_opts_cache;    // for hwdec_interop
};

static const struct libmpv_gpu_next_context_fns *context_backends[] = {
#if HAVE_GL && defined(PL_HAVE_OPENGL)
    &libmpv_gpu_next_context_gl,
#endif
#if HAVE_D3D11 && defined(PL_HAVE_D3D11)
    &libmpv_gpu_next_context_d3d11,
#endif
    NULL
};

static void load_hwdec_api(void *ctx, struct hwdec_imgfmt_request *params)
{
    struct render_backend *rb = ctx;
    struct priv *p = rb->priv;

    pl_video_load_hwdecs(p->video_engine, params);
}

static int init(struct render_backend *ctx, mpv_render_param *params)
{
    ctx->priv = talloc_zero(NULL, struct priv);
    struct priv *p = ctx->priv;

    // Get the API type from the render parameters.
    char *api = get_mpv_render_param(params, MPV_RENDER_PARAM_API_TYPE, NULL);
    if (!api) {
        MP_ERR(ctx, "API type not specified.\n");
        return MPV_ERROR_INVALID_PARAMETER;
    }

    // Find and initialize the requested API context (e.g., GL, D3D11). This
    // creates the pl_gpu and the pl_renderer used by the video engine.
    for (int n = 0; context_backends[n]; n++) {
        const struct libmpv_gpu_next_context_fns *backend = context_backends[n];
        if (strcmp(backend->api_name, api) == 0) {
            p->context = talloc_zero(p, struct libmpv_gpu_next_context);
            *p->context = (struct libmpv_gpu_next_context){
                .global = ctx->global,
                .log = mp_log_new(p, ctx->log, "gpu-next-ctx"),
                .fns = backend,
            };
            break;
        }
    }
    if (!p->context) {
        MP_ERR(ctx, "Requested API type '%s' is not supported.\n", api);
        return MPV_ERROR_NOT_IMPLEMENTED;
    }
    int err = p->context->fns->init(p->context, params);
    if (err < 0) {
        talloc_free(p->context);
        p->context = NULL;
        return err;
    }

    // Initialize our synchronous libplacebo rendering engine.
    p->video_engine = pl_video_init(ctx->global, ctx->log,
                                    p->context->pl_log, p->context->gpu,
                                    p->context->renderer);
    if (!p->video_engine) {
        p->context->fns->destroy(p->context);
        talloc_free(p->context);
        return MPV_ERROR_VO_INIT_FAILED;
    }

    ctx->hwdec_devs = hwdec_devices_create();
    if (p->context->ra_ctx) {
        // The backend can provide a ra_ctx, which we use to set up the hwdec
        // interop and expose it to the decoder via ctx->hwdec_devs. The
        // interop selection mirrors vo_gpu_next's "hwdec-interop" option.
        struct gl_video_opts *gl_opts =
            (p->gl_opts_cache = m_config_cache_alloc(p, ctx->global,
                                                     &gl_video_conf))->opts;
        hwdec_devices_set_loader(ctx->hwdec_devs, load_hwdec_api, ctx);
        pl_video_init_hwdecs(p->video_engine, p->context->ra_ctx,
                             ctx->hwdec_devs, gl_opts->hwdec_interop);
    }

    ctx->driver_caps = VO_CAP_ROTATE90 | VO_CAP_VFLIP;
    return 0;
}

static bool check_format(struct render_backend *ctx, int imgfmt)
{
    struct priv *p = ctx->priv;

    return pl_video_check_format(p->video_engine, imgfmt);
}

static int set_parameter(struct render_backend *ctx, mpv_render_param param)
{
    // No tunable render parameters are implemented for this backend yet.
    return MPV_ERROR_NOT_IMPLEMENTED;
}

static void reconfig(struct render_backend *ctx, struct mp_image_params *params)
{
    struct priv *p = ctx->priv;

    if (p->video_engine)
        pl_video_reconfig(p->video_engine, params);
}

static void reset(struct render_backend *ctx)
{
    struct priv *p = ctx->priv;

    if (p->video_engine)
        pl_video_reset(p->video_engine);
}

static void update_external(struct render_backend *ctx, struct vo *vo)
{
    struct priv *p = ctx->priv;

    if (p->video_engine)
        pl_video_update_osd(p->video_engine, vo ? vo->osd : NULL);
}

static void resize(struct render_backend *ctx, struct mp_rect *src,
                   struct mp_rect *dst, struct mp_osd_res *osd)
{
    struct priv *p = ctx->priv;

    if (p->video_engine)
        pl_video_resize(p->video_engine, src, dst, osd);
}

static int get_target_size(struct render_backend *ctx, mpv_render_param *params,
                           int *out_w, int *out_h)
{
    struct priv *p = ctx->priv;
    if (!p->context || !p->context->fns) return MPV_ERROR_UNINITIALIZED;

    // Mapping the surface is cheap, better than adding new backend entrypoints.
    pl_tex tex = NULL;
    int err = p->context->fns->wrap_fbo(p->context, params, &tex);
    if (err < 0) return err;
    *out_w = tex->params.w;
    *out_h = tex->params.h;
    return 0;
}

static int render(struct render_backend *ctx, mpv_render_param *params,
                  struct vo_frame *frame)
{
    struct priv *p = ctx->priv;
    if (!p->video_engine) return MPV_ERROR_UNINITIALIZED;

    // Mapping the surface is cheap, better than adding new backend entrypoints.
    pl_tex target_tex = NULL;
    int err = p->context->fns->wrap_fbo(p->context, params, &target_tex);
    if (err < 0) return err;

    pl_video_render(p->video_engine, frame, target_tex);

    if (p->context->fns->done_frame)
        p->context->fns->done_frame(p->context);

    return 0;
}

static struct mp_image *get_image(struct render_backend *ctx, int imgfmt,
                                  int w, int h, int stride_align, int flags)
{
    struct priv *p = ctx->priv;

    return pl_video_get_image(p->video_engine, imgfmt, w, h, stride_align, flags);
}

static void screenshot(struct render_backend *ctx, struct vo_frame *frame,
                       struct voctrl_screenshot *args)
{
    struct priv *p = ctx->priv;

    args->res = NULL;
    if (!p || !p->video_engine)
        return;

    args->res = pl_video_screenshot(p->video_engine, frame);
}

static void perfdata(struct render_backend *ctx,
                     struct voctrl_performance_data *out)
{
    // Per-pass timings are not collected by this backend yet.
}

static void destroy(struct render_backend *ctx)
{
    struct priv *p = ctx->priv;
    if (!p) return;

    // Uninit the engine (and with it the hwdec interop) before the device
    // list, since the interop unregisters itself from it.
    pl_video_uninit(&p->video_engine);
    if (ctx->hwdec_devs) {
        hwdec_devices_set_loader(ctx->hwdec_devs, NULL, NULL);
        hwdec_devices_destroy(ctx->hwdec_devs);
    }
    if (p->context) {
        p->context->fns->destroy(p->context);
        talloc_free(p->context);
    }
    talloc_free(p);
    ctx->priv = NULL;
}

const struct render_backend_fns render_backend_gpu_next = {
    .init = init,
    .check_format = check_format,
    .set_parameter = set_parameter,
    .reconfig = reconfig,
    .reset = reset,
    .update_external = update_external,
    .resize = resize,
    .get_target_size = get_target_size,
    .render = render,
    .get_image = get_image,
    .screenshot = screenshot,
    .perfdata = perfdata,
    .destroy = destroy,
};
