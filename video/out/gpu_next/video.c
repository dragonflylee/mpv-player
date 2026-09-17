#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <libplacebo/colorspace.h>
#include <libplacebo/filters.h>
#include <libplacebo/gpu.h>
#include <libplacebo/renderer.h>
#include <libplacebo/utils/frame_queue.h>
#include <libplacebo/utils/upload.h>

#include "common/common.h"
#include "common/msg.h"
#include "options/m_config.h"
#include "sub/draw_bmp.h"
#include "sub/osd.h"
#include "ta/ta_talloc.h"
#include "video/csputils.h"
#include "video/img_format.h"
#include "video/mp_image.h"
#include "video/out/vo.h"
#include "video/out/gpu_next/vo_gpu_next.h"
#include "video.h"

struct osd_entry {
    pl_tex tex;
    struct pl_overlay_part *parts;
    int num_parts;
};

struct pl_osd_state {
    struct osd_entry entries[MAX_OSD_PARTS];
    struct pl_overlay overlays[MAX_OSD_PARTS];
};

struct pl_video {
    struct mp_log *log;
    pl_gpu gpu;                 // borrowed from the API context layer
    pl_renderer renderer;       // owned by the API context layer
    pl_queue queue;

    uint64_t last_frame_id;     // Avoid pushing duplicate frames into the queue.
    double last_pts;

    struct mp_image_params current_params;
    struct mp_rect current_dst;
    struct osd_state *current_osd_state;

    struct mp_osd_res osd_res;
    struct pl_osd_state osd_state;
    pl_fmt osd_fmt[SUBBITMAP_COUNT];
    pl_tex *sub_tex;
    int num_sub_tex;

    struct mp_csp_equalizer_state *video_eq;

    struct m_config_cache *opts_cache;
    struct gl_next_opts *opts;
};

struct frame_priv {
    struct pl_video *p;
};

static void upload_frame_cleanup(struct pl_video *p, struct pl_frame *frame)
{
    if (!frame)
        return;
    for (int i = 0; i < frame->num_planes; i++)
        pl_tex_destroy(p->gpu, &frame->planes[i].texture);
    frame->num_planes = 0;
}

/* Upload an mp_image into a pl_frame. The caller owns the resulting textures
 * and must release them with upload_frame_cleanup(). */
static bool upload_frame(struct pl_video *p, struct pl_frame *out_frame,
                         const struct mp_image *img)
{
    *out_frame = (struct pl_frame){
        .color = img->params.color,
        .repr = img->params.repr,
        .crop = { .x0 = 0, .y0 = 0, .x1 = img->w, .y1 = img->h },
    };

    struct pl_plane_data data[4] = {0};
    int planes = plane_data_from_imgfmt(data, &out_frame->repr.bits,
                                        img->imgfmt, false);
    if (!planes) {
        MP_ERR(p, "Failed to describe image format '%s'\n",
               mp_imgfmt_to_name(img->imgfmt));
        return false;
    }

    out_frame->num_planes = planes;

    // pl_upload_plane() does not modify the mp_image; the cast drops const
    // only because the plane helpers take a non-const pointer.
    struct mp_image *mpi = (struct mp_image *)img;
    for (int n = 0; n < planes; n++) {
        data[n].width = mp_image_plane_w(mpi, n);
        data[n].height = mp_image_plane_h(mpi, n);
        data[n].row_stride = img->stride[n];
        data[n].pixels = img->planes[n];

        if (!pl_upload_plane(p->gpu, &out_frame->planes[n],
                             &out_frame->planes[n].texture, &data[n])) {
            MP_ERR(p, "Failed to upload mp_image plane %d\n", n);
            upload_frame_cleanup(p, out_frame);
            return false;
        }
    }

    pl_frame_set_chroma_location(out_frame, img->params.chroma_location);
    return true;
}

static bool map_frame(pl_gpu gpu, pl_tex *tex, const struct pl_source_frame *src,
                      struct pl_frame *frame)
{
    struct mp_image *mpi = src->frame_data;
    struct pl_video *p = ((struct frame_priv *)mpi->priv)->p;

    if (!upload_frame(p, frame, mpi)) {
        talloc_free(mpi);
        return false;
    }

    frame->user_data = mpi;
    return true;
}

static void unmap_frame(pl_gpu gpu, struct pl_frame *frame,
                        const struct pl_source_frame *src)
{
    struct mp_image *mpi = src->frame_data;
    struct pl_video *p = ((struct frame_priv *)mpi->priv)->p;

    upload_frame_cleanup(p, frame);
    talloc_free(mpi);
}

static void discard_frame(const struct pl_source_frame *src)
{
    talloc_free(src->frame_data);
}

struct pl_video *pl_video_init(struct mpv_global *global, struct mp_log *log,
                               pl_gpu gpu, pl_renderer renderer)
{
    struct pl_video *p = talloc_zero(NULL, struct pl_video);
    p->log = log;
    p->gpu = gpu;
    p->renderer = renderer;
    p->queue = pl_queue_create(gpu);

    // Pre-find the texture formats we'll need for OSD bitmaps for efficiency.
    p->osd_fmt[SUBBITMAP_LIBASS] = pl_find_fmt(gpu, PL_FMT_UNORM, 1, 8, 8, 0);
    p->osd_fmt[SUBBITMAP_BGRA]   = pl_find_fmt(gpu, PL_FMT_UNORM, 4, 8, 8, 0);

    p->video_eq = mp_csp_equalizer_create(p, global);

    p->opts_cache = m_config_cache_alloc(p, global, &gl_next_conf);
    p->opts = p->opts_cache->opts;
    return p;
}

void pl_video_uninit(struct pl_video **p_ptr)
{
    struct pl_video *p = *p_ptr;
    if (!p)
        return;

    pl_queue_destroy(&p->queue);

    for (int i = 0; i < MAX_OSD_PARTS; i++) {
        struct osd_entry *entry = &p->osd_state.entries[i];
        pl_tex_destroy(p->gpu, &entry->tex);
        talloc_free(entry->parts);
    }
    for (int i = 0; i < p->num_sub_tex; i++)
        pl_tex_destroy(p->gpu, &p->sub_tex[i]);
    talloc_free(p->sub_tex);

    talloc_free(p);
    *p_ptr = NULL;
}

static void update_overlays(struct pl_video *p, struct mp_osd_res res,
                            int flags, enum pl_overlay_coords coords,
                            struct pl_osd_state *state, struct pl_frame *frame,
                            struct mp_image *src)
{
    frame->num_overlays = 0;
    if (!p->current_osd_state)
        return;

    // Return OSD textures from the previous frame to the pool so that they can
    // be reused (and resized in place) for the new frame instead of being
    // recreated from scratch every time.
    for (int i = 0; i < MAX_OSD_PARTS; i++) {
        struct osd_entry *entry = &state->entries[i];
        if (entry->tex)
            MP_TARRAY_APPEND(p, p->sub_tex, p->num_sub_tex, entry->tex);
        entry->tex = NULL;
    }

    // Render the logical OSD state into a list of bitmaps.
    double pts = src ? src->pts : 0;
    struct sub_bitmap_list *subs = osd_render(p->current_osd_state, res, pts, flags, mp_draw_sub_formats);
    if (!subs) return;

    frame->overlays = state->overlays;

    // Iterate through each bitmap and convert it into a libplacebo overlay.
    for (int n = 0; n < subs->num_items; n++) {
        const struct sub_bitmaps *item = subs->items[n];
        if (!item->num_parts || !item->packed)
            continue;

        struct osd_entry *entry = &state->entries[item->render_index];
        pl_fmt tex_fmt = p->osd_fmt[item->format];
        if (!entry->tex)
            MP_TARRAY_POP(p->sub_tex, p->num_sub_tex, &entry->tex);
        bool ok = pl_tex_recreate(p->gpu, &entry->tex, &(struct pl_tex_params) {
            .format = tex_fmt,
            .w = MPMAX(item->packed_w, entry->tex ? entry->tex->params.w : 0),
            .h = MPMAX(item->packed_h, entry->tex ? entry->tex->params.h : 0),
            .host_writable = true,
            .sampleable = true,
        });
        if (!ok) {
            MP_ERR(p, "Failed recreating OSD texture!\n");
            break;
        }

        // Upload the new bitmap data to the GPU texture.
        ok = pl_tex_upload(p->gpu, &(struct pl_tex_transfer_params) {
            .tex        = entry->tex,
            .rc         = { .x1 = item->packed_w, .y1 = item->packed_h, },
            .row_pitch  = item->packed->stride[0],
            .ptr        = item->packed->planes[0],
        });
        if (!ok) {
            MP_ERR(p, "Failed uploading OSD texture!\n");
            break;
        }

        entry->num_parts = 0;
        talloc_free(entry->parts);
        entry->parts = talloc_array(p, struct pl_overlay_part, item->num_parts);

        // Convert each sub-bitmap part into a pl_overlay_part.
        for (int i = 0; i < item->num_parts; i++) {
            const struct sub_bitmap *b = &item->parts[i];
            if (b->dw == 0 || b->dh == 0)
                continue;
            uint32_t c = b->libass.color;
            struct pl_overlay_part part = {
                .src = { b->src_x, b->src_y, b->src_x + b->w, b->src_y + b->h },
                .dst = { b->x, b->y, b->x + b->dw, b->y + b->dh },
                .color = {
                    (c >> 24) / 255.0f,
                    ((c >> 16) & 0xFF) / 255.0f,
                    ((c >> 8) & 0xFF) / 255.0f,
                    (255 - (c & 0xFF)) / 255.0f,
                }
            };
            entry->parts[entry->num_parts++] = part;
        }

        // Create the final pl_overlay structure for rendering.
        struct pl_overlay *ol = &state->overlays[frame->num_overlays++];
        *ol = (struct pl_overlay) {
            .tex = entry->tex,
            .parts = entry->parts,
            .num_parts = entry->num_parts,
            .color = { .primaries = PL_COLOR_PRIM_BT_709, .transfer = PL_COLOR_TRC_SRGB },
            .coords = coords,
        };

        // Set blending modes based on the OSD bitmap format.
        switch (item->format) {
        case SUBBITMAP_BGRA:
            ol->mode = PL_OVERLAY_NORMAL;
            ol->repr.alpha = PL_ALPHA_PREMULTIPLIED;
            // Infer bitmap colorspace from source
            if (src) {
                ol->color = src->params.color;
                if (pl_color_transfer_is_hdr(ol->color.transfer)) {
                    if (!pl_color_transfer_is_hdr(frame->color.transfer)) {
                        // Tone mapping targets SDR white
                        ol->color.hdr = (struct pl_hdr_metadata) {
                            .max_luma = PL_COLOR_SDR_WHITE,
                        };
                    } else if (p->opts->image_subs_hdr_peak != -1) {
                        ol->color.hdr = (struct pl_hdr_metadata) {
                            .max_luma = p->opts->image_subs_hdr_peak,
                        };
                    }
                }
            }
            break;
        case SUBBITMAP_LIBASS:
            if (src && item->video_color_space &&
                !pl_color_space_is_hdr(&src->params.color))
                ol->color = src->params.color;
            if (src && pl_color_transfer_is_hdr(frame->color.transfer)) {
                ol->color.hdr = (struct pl_hdr_metadata) {
                    .max_luma = p->opts->sub_hdr_peak,
                };
            }
            ol->mode = PL_OVERLAY_MONOCHROME;
            ol->repr.alpha = PL_ALPHA_INDEPENDENT;
            break;
        }
    }

    talloc_free(subs);
}

void pl_video_render(struct pl_video *p, struct vo_frame *frame, pl_tex target_tex)
{
    struct pl_frame target = {
        .num_planes = 1,
        .planes[0] = { .texture = target_tex, .components = 4, .component_mapping = {0,1,2,3} },
        .crop = { .x0 = p->current_dst.x0, .y0 = p->current_dst.y0,
                  .x1 = p->current_dst.x1, .y1 = p->current_dst.y1 },
        .color = pl_color_space_srgb,
        .repr = pl_color_repr_rgb,
    };

    // libmpv hands us one new frame at a time in frame->current; frame_id
    // guards against pushing the same frame twice.
    if (frame && frame->current && frame->frame_id > p->last_frame_id) {
        struct mp_image *mpi = mp_image_new_ref(frame->current);
        struct frame_priv *fp = talloc_zero(mpi, struct frame_priv);
        fp->p = p;
        mpi->priv = fp;

        pl_queue_push(p->queue, &(struct pl_source_frame) {
            .pts = mpi->pts,
            .frame_data = mpi,
            .map = map_frame,
            .unmap = unmap_frame,
            .discard = discard_frame,
        });

        p->last_frame_id = frame->frame_id;
    }

    // On a redraw frame->current is NULL; reuse the last PTS.
    double pts = (frame && frame->current) ? frame->current->pts : p->last_pts;
    p->last_pts = pts;

    struct pl_frame_mix mix = {0};
    pl_queue_update(p->queue, &mix, pl_queue_params(.pts = pts));

    // pl_frame_mix.signatures is only valid for the duration of pl_queue_update;
    // point it at a local array using the frame pointer as the cache signature.
    uint64_t signatures[32];
    assert(mix.num_frames <= MP_ARRAY_SIZE(signatures));
    for (int i = 0; i < mix.num_frames; i++)
        signatures[i] = (uintptr_t)mix.frames[i]->user_data;
    mix.signatures = signatures;

    // The first frame in the mix provides the colorspace for the OSD. It is
    // NULL when there is no video, in which case overlays render on black.
    struct mp_image *ref = mix.num_frames > 0 ? mix.frames[0]->user_data : NULL;
    update_overlays(p, p->osd_res, 0, PL_OVERLAY_COORDS_DST_FRAME,
                    &p->osd_state, &target, ref);

    // No interpolation; assume the frame is displayed for one vsync.
    mix.vsync_duration = 1.0f;

    struct pl_color_adjustment color_adj;
    struct mp_csp_params cparams = MP_CSP_PARAMS_DEFAULTS;
    mp_csp_equalizer_state_get(p->video_eq, &cparams);
    color_adj.brightness = cparams.brightness;
    color_adj.contrast   = cparams.contrast;
    color_adj.hue        = cparams.hue;
    color_adj.saturation = cparams.saturation;
    color_adj.gamma      = cparams.gamma;

    struct pl_render_params params = pl_render_default_params;
    params.upscaler = &pl_filter_ewa_lanczossharp;
    params.downscaler = &pl_filter_ewa_lanczos;
    params.color_adjustment = &color_adj;

    if (!pl_render_image_mix(p->renderer, &mix, &target, &params))
        MP_ERR(p, "Rendering failed.\n");
}

struct mp_image *pl_video_screenshot(struct pl_video *p, struct vo_frame *frame)
{
    if (!p || !frame || !frame->current)
        return NULL;

    struct mp_image *res = NULL;
    struct pl_frame source = {0};
    pl_tex fbo = NULL;

    if (!upload_frame(p, &source, frame->current)) {
        MP_ERR(p, "Failed to upload source image for screenshot.\n");
        return NULL;
    }

    int w = frame->current->w, h = frame->current->h;
    pl_fmt fmt = pl_find_fmt(p->gpu, PL_FMT_UNORM, 4, 8, 8,
                             PL_FMT_CAP_RENDERABLE | PL_FMT_CAP_HOST_READABLE);
    if (!fmt) {
        MP_ERR(p, "Failed to find screenshot format.\n");
        goto done;
    }

    fbo = pl_tex_create(p->gpu, pl_tex_params(
        .w = w,
        .h = h,
        .format = fmt,
        .renderable = true,
        .host_readable = true
    ));
    if (!fbo) {
        MP_ERR(p, "Failed to create screenshot texture.\n");
        goto done;
    }

    // Target sRGB SDR, so libplacebo performs the tone-mapping.
    struct pl_frame target = {
        .num_planes = 1,
        .planes[0] = { .texture = fbo, .components = 4, .component_mapping = {0,1,2,3} },
        .color = pl_color_space_srgb,
        .repr = pl_color_repr_rgb,
    };

    struct mp_osd_res osd = {
        .w = w,
        .h = h,
        .display_par = 1.0, // Screenshots have square pixels
    };
    update_overlays(p, osd, 0, PL_OVERLAY_COORDS_DST_FRAME, &p->osd_state, &target,
                    frame->current);

    if (!pl_render_image(p->renderer, &source, &target, &pl_render_default_params)) {
        MP_ERR(p, "Screenshot rendering failed.\n");
        goto done;
    }

    res = mp_image_alloc(IMGFMT_RGBA, w, h);
    if (!res) {
        MP_ERR(p, "Failed to allocate screenshot image.\n");
        goto done;
    }

    if (!pl_tex_download(p->gpu, &(struct pl_tex_transfer_params){
        .tex = fbo,
        .ptr = res->planes[0],
        .row_pitch = res->stride[0],
    })) {
        MP_ERR(p, "Screenshot texture download failed.\n");
        talloc_free(res);
        res = NULL;
    }

done:
    pl_tex_destroy(p->gpu, &fbo);
    upload_frame_cleanup(p, &source);
    return res;
}

void pl_video_reconfig(struct pl_video *p, const struct mp_image_params *params)
{
    if (params)
        p->current_params = *params;
    m_config_cache_update(p->opts_cache);
}

void pl_video_resize(struct pl_video *p, const struct mp_rect *dst,
                     const struct mp_osd_res *osd)
{
    if (dst)
        p->current_dst = *dst;
    if (osd)
        p->osd_res = *osd;
}

void pl_video_update_osd(struct pl_video *p, struct osd_state *osd)
{
    p->current_osd_state = osd;
}

void pl_video_reset(struct pl_video *p)
{
    pl_renderer_flush_cache(p->renderer);
    pl_queue_reset(p->queue);
    p->last_frame_id = 0;
    p->last_pts = 0;
}

