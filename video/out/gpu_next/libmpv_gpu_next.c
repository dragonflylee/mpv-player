#include "config.h"
#include "common/msg.h"
#include "libmpv_gpu_next.h"
#include "mpv/client.h"
#include "mpv/render.h"
#include "ta/ta_talloc.h"
#include "video/hwdec.h"
#include "video/out/libmpv.h"
#include "video/out/vo.h"
#include "video.h"

struct priv {
    struct libmpv_gpu_next_context *context; // Manages the API (e.g., GL, D3D11)
    struct pl_video *video_engine;           // Manages synchronous libplacebo rendering
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
                                    p->context->gpu, p->context->renderer);
    if (!p->video_engine) {
        p->context->fns->destroy(p->context);
        talloc_free(p->context);
        return MPV_ERROR_VO_INIT_FAILED;
    }

    ctx->hwdec_devs = hwdec_devices_create();
    ctx->driver_caps = VO_CAP_ROTATE90 | VO_CAP_VFLIP;
    return 0;
}

static bool check_format(struct render_backend *ctx, int imgfmt)
{
    // libplacebo handles the conversion; accept any format it can upload.
    return true;
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
        pl_video_resize(p->video_engine, dst, osd);
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
    // No zero-copy direct rendering support; the core falls back to normal
    // frames provided through vo_frame.
    return NULL;
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

    hwdec_devices_destroy(ctx->hwdec_devs);
    pl_video_uninit(&p->video_engine);
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
