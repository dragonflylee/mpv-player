#pragma once

#include <stdbool.h>
#include <libplacebo/gpu.h>
#include <libplacebo/renderer.h>

struct mp_image;
struct mp_image_params;
struct mp_log;
struct mp_osd_res;
struct mp_rect;
struct mpv_global;
struct mp_hwdec_devices;
struct hwdec_imgfmt_request;
struct osd_state;
struct ra_ctx;
struct vo_frame;

// Synchronous libplacebo rendering engine used by the 'gpu-next' libmpv
// render backend. pllog/gpu/renderer are borrowed from the API context layer
// and must outlive the engine.
struct pl_video *pl_video_init(struct mpv_global *global, struct mp_log *log,
                               pl_log pllog, pl_gpu gpu, pl_renderer renderer);
void pl_video_uninit(struct pl_video **p_ptr);

void pl_video_render(struct pl_video *p, struct vo_frame *frame, pl_tex target_tex);

// Render frame into a temporary sRGB image and return it as a newly allocated
// RGBA mp_image, or NULL on failure.
struct mp_image *pl_video_screenshot(struct pl_video *p, struct vo_frame *frame);

// Enable hardware decoding interop. ra_ctx is the anchor for the interop and
// must outlive the engine; interop selects the drivers, matching the
// "hwdec-interop" option semantics. All matching drivers are loaded eagerly so
// that check_format() can report the hwdec formats.
void pl_video_init_hwdecs(struct pl_video *p, struct ra_ctx *ra_ctx,
                          struct mp_hwdec_devices *devs, const char *interop);
void pl_video_load_hwdecs(struct pl_video *p, struct hwdec_imgfmt_request *params);

// Whether the format can be rendered, either as a hwdec surface (mapped
// zero-copy) or as an uploadable swdec format.
bool pl_video_check_format(struct pl_video *p, int imgfmt);

// Allocate a host-mapped buffer that the decoder can write into directly, for
// use as struct render_backend_fns.get_image.
struct mp_image *pl_video_get_image(struct pl_video *p, int imgfmt, int w, int h,
                                    int stride_align, int flags);

void pl_video_reconfig(struct pl_video *p, const struct mp_image_params *params);
void pl_video_resize(struct pl_video *p, const struct mp_rect *src,
                     const struct mp_rect *dst, const struct mp_osd_res *osd);
void pl_video_update_osd(struct pl_video *p, struct osd_state *osd);
void pl_video_reset(struct pl_video *p);

