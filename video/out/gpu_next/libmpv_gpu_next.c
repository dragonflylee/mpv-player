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

#include <stdint.h>

#include <libplacebo/colorspace.h>
#include <libplacebo/gpu.h>
#include <libplacebo/options.h>
#include <libplacebo/renderer.h>
#include <libplacebo/utils/frame_queue.h>
#include <libplacebo/utils/upload.h>

#include "config.h"
#include "common/common.h"
#include "common/msg.h"
#include "libmpv_gpu_next.h"
#include "options/m_config.h"
#include "sub/draw_bmp.h"
#include "sub/osd.h"
#include "ta/ta_talloc.h"
#include "video/csputils.h"
#include "video/img_format.h"
#include "video/mp_image.h"
#include "video/out/gpu/hwdec.h"
#include "video/out/gpu/video.h"
#include "video/out/gpu_next/vo_gpu_next.h"
#include "video/out/libmpv.h"
#include "video/out/placebo/ra_pl.h"
#include "video/out/vo.h"

#if HAVE_GL && defined(PL_HAVE_OPENGL)
#include <libplacebo/opengl.h>
#include "video/out/opengl/ra_gl.h"
#endif

#if HAVE_D3D11 && defined(PL_HAVE_D3D11)
#include <libplacebo/d3d11.h>
#include "video/out/d3d11/ra_d3d11.h"
#include "osdep/windows_utils.h"
#endif

// Upper bound for the host-mapped DR buffer pool.
#define MP_MAX_DR_BUFFERS 128

struct osd_entry {
    pl_tex tex;
    struct pl_overlay_part *parts;
    int num_parts;
    int change_id;
};

struct osd_state {
    struct osd_entry entries[MAX_OSD_PARTS];
    struct pl_overlay *overlays;
    int num_overlays;
};

// Mapping state of single hwdec frame.
struct hwdec_slot {
    struct ra_hwdec_mapper *mapper;
    struct mp_image *owner;   // queue entry image that owns the mapping
    bool acquired;            // between acquire and release of a render pass
    pl_tex tex[4];            // plane textures of the current mapping
};

struct priv {
    struct mp_log *log;
    struct mpv_global *global;
    struct libmpv_gpu_next_context *context;
    struct ra_hwdec_ctx hwdec_ctx;
    struct hwdec_slot hwdec;

    // Allocated DR buffers
    mp_mutex dr_lock;
    pl_buf *dr_buffers;
    int num_dr_buffers;

    pl_log pllog;
    pl_gpu gpu;
    pl_renderer rr;
    pl_queue queue;
    pl_fmt osd_fmt[SUBBITMAP_COUNT];
    pl_tex *sub_tex;
    int num_sub_tex;

    struct mp_rect src, dst;
    struct mp_osd_res osd_res;
    struct osd_state osd_state;

    // OSD state queried from the frontend via update_external().
    struct osd_state *current_osd;

    uint64_t last_id;
    uint64_t osd_sync;
    double last_pts;
    bool want_reset;
    bool flush_cache;

    pl_options pars;
    struct m_config_cache *opts_cache;
    struct gl_next_opts *next_opts;
    struct m_config_cache *gl_opts_cache;
    struct mp_csp_equalizer_state *video_eq;
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

struct frame_priv {
    struct priv *p;
    struct ra_hwdec *hwdec;
    struct osd_state subs;
    uint64_t osd_sync;
    uint64_t signature;         // unique per queue entry, see draw_frame()
};

static pl_buf get_dr_buf(struct priv *p, const uint8_t *ptr)
{
    mp_mutex_lock(&p->dr_lock);

    for (int i = 0; i < p->num_dr_buffers; i++) {
        pl_buf buf = p->dr_buffers[i];
        if (ptr >= buf->data && ptr < buf->data + buf->params.size) {
            mp_mutex_unlock(&p->dr_lock);
            return buf;
        }
    }

    mp_mutex_unlock(&p->dr_lock);
    return NULL;
}

static void free_dr_buf(void *opaque, uint8_t *data)
{
    struct priv *p = opaque;
    mp_mutex_lock(&p->dr_lock);

    if (!p->gpu) {
        mp_mutex_unlock(&p->dr_lock);
        return;
    }

    for (int i = 0; i < p->num_dr_buffers; i++) {
        if (p->dr_buffers[i]->data == data) {
            pl_buf_destroy(p->gpu, &p->dr_buffers[i]);
            MP_TARRAY_REMOVE_AT(p->dr_buffers, p->num_dr_buffers, i);
            mp_mutex_unlock(&p->dr_lock);
            return;
        }
    }

    mp_mutex_unlock(&p->dr_lock);
    MP_ASSERT_UNREACHABLE();
}

static struct mp_image *get_image(struct render_backend *ctx, int imgfmt,
                                  int w, int h, int stride_align, int flags)
{
    struct priv *p = ctx->priv;
    pl_gpu gpu = p->gpu;
    if (!gpu->limits.thread_safe || !gpu->limits.max_mapped_size)
        return NULL;

    if ((flags & VO_DR_FLAG_HOST_CACHED) && !gpu->limits.host_cached)
        return NULL;

    stride_align = mp_lcm(stride_align, gpu->limits.align_tex_xfer_pitch);
    stride_align = mp_lcm(stride_align, gpu->limits.align_tex_xfer_offset);
    int size = mp_image_get_alloc_size(imgfmt, w, h, stride_align);
    if (size < 0)
        return NULL;

    pl_buf buf = pl_buf_create(gpu, &(struct pl_buf_params) {
        .memory_type = PL_BUF_MEM_HOST,
        .host_mapped = true,
        .size = size + stride_align,
    });

    if (!buf)
        return NULL;

    struct mp_image *mpi = mp_image_from_buffer(imgfmt, w, h, stride_align,
                                                buf->data, buf->params.size,
                                                p, free_dr_buf);
    if (!mpi) {
        pl_buf_destroy(gpu, &buf);
        return NULL;
    }

    mp_mutex_lock(&p->dr_lock);
    bool ok = p->num_dr_buffers < MP_MAX_DR_BUFFERS;
    if (ok)
        MP_TARRAY_APPEND(p, p->dr_buffers, p->num_dr_buffers, buf);
    mp_mutex_unlock(&p->dr_lock);

    if (!ok) {
        MP_WARN(p, "DR buffer pool exhausted (%d buffers), falling back to "
                   "software decoding.\n", MP_MAX_DR_BUFFERS);
        pl_buf_destroy(gpu, &buf);
        talloc_free(mpi);
        return NULL;
    }

    return mpi;
}

static bool upload_overlay_tex(struct priv *p, struct osd_entry *entry,
                               const struct sub_bitmaps *item)
{
    pl_fmt tex_fmt = p->osd_fmt[item->format];
    if (!entry->tex)
        MP_TARRAY_POP(p->sub_tex, p->num_sub_tex, &entry->tex);
    // pl_tex_recreate() invalidates the texture even when it keeps it, so only
    // call it if the size or format has to change. Textures only ever grow.
    pl_tex tex = entry->tex;
    if (!tex || tex->params.w < item->packed_w ||
        tex->params.h < item->packed_h || tex->params.format != tex_fmt)
    {
        entry->change_id = 0;
        if (!pl_tex_recreate(p->gpu, &entry->tex, &(struct pl_tex_params) {
                .format = tex_fmt,
                .w = MPMAX(item->packed_w, tex ? tex->params.w : 0),
                .h = MPMAX(item->packed_h, tex ? tex->params.h : 0),
                .host_writable = true,
                .sampleable = true,
            }))
        {
            MP_ERR(p, "Failed recreating OSD texture!\n");
            return false;
        }
    }
    if (entry->change_id == item->change_id)
        return true;
    struct pl_tex_transfer_params tp = {
        .tex        = entry->tex,
        .rc         = { .x1 = item->packed_w, .y1 = item->packed_h, },
        .row_pitch  = item->packed->stride[0],
        .ptr        = item->packed->planes[0],
    };
    // Keep the image alive until it's fully read.
    if (p->gpu->limits.callbacks) {
        tp.callback = talloc_free;
        tp.priv = mp_image_new_ref(item->packed);
    }
    if (!pl_tex_upload(p->gpu, &tp)) {
        MP_ERR(p, "Failed uploading OSD texture!\n");
        talloc_free(tp.priv);
        entry->change_id = 0;
        return false;
    }
    entry->change_id = item->change_id;
    return true;
}

static void recycle_subs(struct priv *p, struct osd_state *state)
{
    for (int i = 0; i < MP_ARRAY_SIZE(state->entries); i++) {
        struct osd_entry *entry = &state->entries[i];
        if (entry->tex)
            MP_TARRAY_APPEND(p, p->sub_tex, p->num_sub_tex, entry->tex);
        entry->tex = NULL;
        talloc_free(entry->parts);
        entry->parts = NULL;
    }
}

static void update_overlays(struct priv *p, struct mp_osd_res res, int flags,
                            enum pl_overlay_coords coords,
                            struct osd_state *state, struct pl_frame *frame,
                            struct mp_image *src)
{
    if (!p->current_osd)
        return;

    recycle_subs(p, state);

    double pts = src ? src->pts : 0;
    struct sub_bitmap_list *subs = osd_render(p->current_osd, res, pts,
                                              flags, mp_draw_sub_formats);
    if (!subs)
        return;

    state->num_overlays = 0;

    for (int n = 0; n < subs->num_items; n++) {
        const struct sub_bitmaps *item = subs->items[n];
        if (!item->num_parts || !item->packed)
            continue;

        struct osd_entry *entry = &state->entries[item->render_index];
        if (!upload_overlay_tex(p, entry, item))
            break;

        bool mono = item->format == SUBBITMAP_LIBASS;
        struct pl_color_space color = pl_color_space_srgb;

        if (src) {
            bool src_hdr = pl_color_transfer_is_hdr(src->params.color.transfer);
            if (!mono || (item->video_color_space && !src_hdr))
                color = src->params.color;

            if (mono) {
                if (pl_color_transfer_is_hdr(frame->color.transfer) &&
                    p->next_opts->sub_hdr_peak)
                    color.hdr.max_luma = p->next_opts->sub_hdr_peak;
            } else if (pl_color_transfer_is_hdr(color.transfer)) {
                if (!pl_color_transfer_is_hdr(frame->color.transfer))
                    color.hdr.max_luma = PL_COLOR_SDR_WHITE;
                else if (p->next_opts->image_subs_hdr_peak != -1)
                    color.hdr.max_luma = p->next_opts->image_subs_hdr_peak;
            }
        }

        entry->num_parts = 0;
        talloc_free(entry->parts);
        entry->parts = NULL;

        for (int i = 0; i < item->num_parts; i++) {
            const struct sub_bitmap *b = &item->parts[i];
            if (b->dw == 0 || b->dh == 0)
                continue;

            struct pl_overlay_part part = {
                .src = { b->src_x, b->src_y, b->src_x + b->w, b->src_y + b->h },
                .dst = { b->x, b->y, b->x + b->dw, b->y + b->dh },
            };
            if (mono) {
                uint32_t c = b->libass.color;
                part.color[0] = (c >> 24) / 255.0f;
                part.color[1] = ((c >> 16) & 0xFF) / 255.0f;
                part.color[2] = ((c >> 8) & 0xFF) / 255.0f;
                part.color[3] = (255 - (c & 0xFF)) / 255.0f;
            }
            MP_TARRAY_APPEND(p, entry->parts, entry->num_parts, part);
        }

        MP_TARRAY_GROW(p, state->overlays, state->num_overlays);
        struct pl_overlay *ol = &state->overlays[state->num_overlays++];
        *ol = (struct pl_overlay) {
            .tex = entry->tex,
            .parts = entry->parts,
            .num_parts = entry->num_parts,
            .color = color,
            .coords = coords,
        };

        switch (item->format) {
        case SUBBITMAP_BGRA:
            ol->mode = PL_OVERLAY_NORMAL;
            ol->repr.alpha = PL_ALPHA_PREMULTIPLIED;
            break;
        case SUBBITMAP_LIBASS:
            ol->mode = PL_OVERLAY_MONOCHROME;
            ol->repr.alpha = PL_ALPHA_INDEPENDENT;
            break;
        }
    }

    frame->overlays = state->overlays;
    frame->num_overlays = state->num_overlays;

    talloc_free(subs);
}

static void upload_frame_cleanup(struct priv *p, struct pl_frame *frame)
{
    for (int i = 0; i < frame->num_planes; i++)
        pl_tex_destroy(p->gpu, &frame->planes[i].texture);
    frame->num_planes = 0;
}

static void slot_unmap(struct priv *p, struct hwdec_slot *s)
{
    mp_assert(!s->acquired);
    if (!s->mapper)
        return;
    // For RAs not based on ra_pl the plane textures are wrappers we created.
    if (!ra_pl_get(s->mapper->ra)) {
        for (int n = 0; n < MP_ARRAY_SIZE(s->tex); n++)
            pl_tex_destroy(p->gpu, &s->tex[n]);
    }
    memset(s->tex, 0, sizeof(s->tex));
    ra_hwdec_mapper_unmap(s->mapper);
    s->owner = NULL;
}

static bool slot_reconfig(struct priv *p, struct hwdec_slot *s,
                          struct ra_hwdec *hwdec,
                          const struct mp_image_params *par)
{
    if (s->mapper) {
        if (mp_image_params_static_equal(par, &s->mapper->src_params)) {
            s->mapper->src_params.repr.dovi = par->repr.dovi;
            s->mapper->dst_params.repr.dovi = par->repr.dovi;
            s->mapper->src_params.color.hdr = par->color.hdr;
            s->mapper->dst_params.color.hdr = par->color.hdr;
            return true;
        }
        slot_unmap(p, s);
        ra_hwdec_mapper_free(&s->mapper);
    }

    s->mapper = ra_hwdec_mapper_create(hwdec, par);
    if (!s->mapper) {
        MP_ERR(p, "Initializing texture for hardware decoding failed.\n");
        return false;
    }

    return true;
}

// The queue dropped this entry. Release the mapping if the entry owns it.
static void slot_release_owner(struct priv *p, struct hwdec_slot *s,
                               struct mp_image *owner)
{
    if (s->owner == owner)
        slot_unmap(p, s);
}

static void slot_uninit(struct priv *p, struct hwdec_slot *s)
{
    slot_unmap(p, s);
    ra_hwdec_mapper_free(&s->mapper);
}

// For RAs not based on ra_pl, this creates a new pl_tex wrapper.
static pl_tex hwdec_get_tex(struct priv *p, struct ra_hwdec_mapper *mapper, int n)
{
    struct ra_tex *ratex = mapper->tex[n];
    struct ra *ra = mapper->ra;
    if (ra_pl_get(ra))
        return (pl_tex) ratex->priv;

#if HAVE_GL && defined(PL_HAVE_OPENGL)
    if (ra_is_gl(ra) && pl_opengl_get(p->gpu)) {
        struct pl_opengl_wrap_params par = {
            .width = ratex->params.w,
            .height = ratex->params.h,
        };

        ra_gl_get_format(ratex->params.format, &par.iformat,
                         &(GLenum){0}, &(GLenum){0});
        ra_gl_get_raw_tex(ra, ratex, &par.texture, &par.target);
        return pl_opengl_wrap(p->gpu, &par);
    }
#endif

#if HAVE_D3D11 && defined(PL_HAVE_D3D11)
    if (ra_is_d3d11(ra)) {
        int array_slice = 0;
        ID3D11Resource *res = ra_d3d11_get_raw_tex(ra, ratex, &array_slice);
        pl_tex tex = pl_d3d11_wrap(p->gpu, pl_d3d11_wrap_params(
            .tex = res,
            .array_slice = array_slice,
            .fmt = ra_d3d11_get_format(ratex->params.format),
            .w = ratex->params.w,
            .h = ratex->params.h,
        ));
        SAFE_RELEASE(res);
        return tex;
    }
#endif

    MP_ERR(p, "Failed mapping hwdec frame? Open a bug!\n");
    return NULL;
}

// Fill `frame->num_planes` and per-plane component_mapping from an
// hwdec-mapped imgfmt description.
static void setup_hwdec_plane_mapping(struct pl_frame *frame,
                                      struct mp_imgfmt_desc *desc)
{
    frame->num_planes = desc->num_planes;
    for (int n = 0; n < frame->num_planes; n++) {
        struct pl_plane *plane = &frame->planes[n];
        int *map = plane->component_mapping;
        for (int c = 0; c < mp_imgfmt_desc_get_num_comps(desc); c++) {
            if (desc->comps[c].plane != n)
                continue;
            // Sort by component offset
            uint8_t offset = desc->comps[c].offset;
            int index = plane->components++;
            while (index > 0 && desc->comps[map[index - 1]].offset > offset) {
                map[index] = map[index - 1];
                index--;
            }
            map[index] = c;
        }
    }
}

static bool slot_acquire(struct priv *p, struct hwdec_slot *s,
                         struct mp_image *owner, struct mp_image *mpi,
                         struct pl_frame *frame)
{
    if (s->owner != owner) {
        // With one frame per role acquired at a time the previous owner is
        // not in use and can be evicted. Interlaced sources would break this.
        if (s->acquired) {
            MP_ERR(p, "Hardware frame of this role is still in use.\n");
            return false;
        }
        slot_unmap(p, s);

        int ret = ra_hwdec_mapper_map(s->mapper, mpi);
        if (ret < 0) {
            MP_ERR(p, "Mapping hardware decoded surface failed.\n");
            return false;
        }
        s->owner = owner;
    }

    // Set before handing out textures so that the release libplacebo issues
    // after a failed acquire unmaps the frame again.
    s->acquired = true;

    for (int n = 0; n < frame->num_planes; n++) {
        if (!s->tex[n])
            s->tex[n] = hwdec_get_tex(p, s->mapper, n);
        if (!s->tex[n])
            return false;
        frame->planes[n].texture = s->tex[n];
    }

    return true;
}

static void slot_release(struct priv *p, struct hwdec_slot *s)
{
    if (!s->acquired)
        return;
    s->acquired = false;
    slot_unmap(p, s);
}

static bool hwdec_acquire(pl_gpu gpu, struct pl_frame *frame)
{
    struct mp_image *mpi = frame->user_data;
    struct frame_priv *fp = mpi->priv;
    struct priv *p = fp->p;
    return slot_reconfig(p, &p->hwdec, fp->hwdec, &mpi->params) &&
           slot_acquire(p, &p->hwdec, mpi, mpi, frame);
}

static void hwdec_release(pl_gpu gpu, struct pl_frame *frame)
{
    struct mp_image *mpi = frame->user_data;
    struct frame_priv *fp = mpi->priv;
    struct priv *p = fp->p;
    slot_release(p, &p->hwdec);
}

static bool format_supported(struct priv *p, int format, bool use_uint)
{
    struct pl_bit_encoding bits;
    struct pl_plane_data data[4] = {0};
    int planes = plane_data_from_imgfmt(data, &bits, format, use_uint);
    if (!planes)
        return false;

    for (int i = 0; i < planes; i++) {
        if (!pl_plane_find_fmt(p->gpu, NULL, &data[i]))
            return false;
    }

    return true;
}

static bool upload_planes_sw(struct priv *p, pl_gpu gpu, struct mp_image *mpi,
                             struct pl_frame *frame, pl_tex tex[4])
{
    struct pl_plane_data data[4] = {0};

    // At this point, we know that the format is supported, check_format()
    // makes sure of that. Just check if we should use UINT as a fallback.
    bool use_uint = !format_supported(p, mpi->imgfmt, false);
    int planes = plane_data_from_imgfmt(data, &frame->repr.bits, mpi->imgfmt,
                                        use_uint);
    if (!planes)
        return false;

    frame->num_planes = planes;
    for (int n = 0; n < planes; n++) {
        struct pl_plane *plane = &frame->planes[n];
        data[n].width = mp_image_plane_w(mpi, n);
        data[n].height = mp_image_plane_h(mpi, n);
        if (mpi->stride[n] < 0) {
            data[n].pixels = mpi->planes[n] + (data[n].height - 1) * mpi->stride[n];
            data[n].row_stride = -mpi->stride[n];
            plane->flipped = true;
        } else {
            data[n].pixels = mpi->planes[n];
            data[n].row_stride = mpi->stride[n];
        }

        pl_buf buf = get_dr_buf(p, data[n].pixels);
        if (buf) {
            data[n].buf = buf;
            data[n].buf_offset = (uint8_t *) data[n].pixels - buf->data;
            data[n].pixels = NULL;
        }
        // Keep the image alive until it's fully read.
        if (gpu->limits.callbacks) {
            data[n].callback = talloc_free;
            data[n].priv = mp_image_new_ref(mpi);
        }

        if (!pl_upload_plane(gpu, plane, &tex[n], &data[n])) {
            talloc_free(data[n].priv);
            return false;
        }
        plane->texture = tex[n];

        // Without async callback support, we have to poll...
        if (!gpu->limits.callbacks && data[n].buf)
            while (pl_buf_poll(gpu, data[n].buf, UINT64_MAX));
    }

    return true;
}

// Upload a single frame outside the queue, for screenshots. The caller must
// call upload_frame_cleanup() afterwards.
static bool upload_frame(struct priv *p, struct pl_frame *out_frame,
                         struct mp_image *img)
{
    *out_frame = (struct pl_frame) {
        .color = img->params.color,
        .repr = img->params.repr,
    };

    pl_tex tex[4] = {0};
    bool ok = upload_planes_sw(p, p->gpu, img, out_frame, tex);
    if (!ok)
        upload_frame_cleanup(p, out_frame);
    return ok;
}

static bool map_frame(pl_gpu gpu, pl_tex *tex, const struct pl_source_frame *src,
                      struct pl_frame *frame)
{
    struct mp_image *mpi = src->frame_data;
    struct mp_image_params par = mpi->params;
    struct frame_priv *fp = mpi->priv;
    struct priv *p = fp->p;

    fp->hwdec = p->hwdec_ctx.ra_ctx
        ? ra_hwdec_get(&p->hwdec_ctx, mpi->imgfmt) : NULL;
    if (fp->hwdec) {
        // Only reconfig the mapper here to access `dst_params`; the actual
        // mapping happens in acquire().
        if (!slot_reconfig(p, &p->hwdec, fp->hwdec, &mpi->params))
            return false;

        par = p->hwdec.mapper->dst_params;
    }

    mp_image_params_guess_csp(&par);

    *frame = (struct pl_frame) {
        .color = par.color,
        .repr = par.repr,
        .profile = {
            .data = mpi->icc_profile ? mpi->icc_profile->data : NULL,
            .len = mpi->icc_profile ? mpi->icc_profile->size : 0,
        },
        .rotation = par.rotate / 90,
        .user_data = mpi,
    };

    // Honor the reference white override and the sRGB->gamma22 knob, which are
    // baked into the mapped frame (see reconfig()).
    const struct gl_video_opts *opts = p->gl_opts_cache->opts;
    if (!pl_color_transfer_is_hdr(frame->color.transfer) && opts->hdr_reference_white)
        frame->color.hdr.max_luma = opts->hdr_reference_white;

    if ((opts->treat_srgb_as_power22 & 1) &&
        frame->color.transfer == PL_COLOR_TRC_SRGB)
    {
        // The sRGB EOTF is a pure gamma 2.2 function.
        frame->color.transfer = PL_COLOR_TRC_GAMMA22;
    }

    if (fp->hwdec) {
        struct mp_imgfmt_desc desc = mp_imgfmt_get_desc(par.imgfmt);
        frame->acquire = hwdec_acquire;
        frame->release = hwdec_release;
        setup_hwdec_plane_mapping(frame, &desc);
    } else { // swdec
        if (!upload_planes_sw(p, gpu, mpi, frame, tex)) {
            MP_ERR(p, "Failed uploading frame!\n");
            upload_frame_cleanup(p, frame);
            return false;
        }
    }

    // Update chroma location, must be done after initializing planes
    pl_frame_set_chroma_location(frame, par.chroma_location);

    pl_icc_profile_compute_signature(&frame->profile);
    return true;
}

static void unmap_frame(pl_gpu gpu, struct pl_frame *frame,
                        const struct pl_source_frame *src)
{
    struct mp_image *mpi = src->frame_data;
    struct frame_priv *fp = mpi->priv;
    struct priv *p = fp->p;

    if (!fp->hwdec)
        upload_frame_cleanup(p, frame);

    recycle_subs(p, &fp->subs);
    slot_release_owner(p, &p->hwdec, mpi);
    talloc_free(mpi);
}

static void discard_frame(const struct pl_source_frame *src)
{
    talloc_free(src->frame_data);
}

static void update_options(struct render_backend *ctx)
{
    struct priv *p = ctx->priv;
    pl_options pars = p->pars;

    bool changed = m_config_cache_update(p->opts_cache);
    changed = m_config_cache_update(p->gl_opts_cache) || changed;
    if (changed)
        p->want_reset = true;

    // Update equalizer state
    struct mp_csp_params cparams = MP_CSP_PARAMS_DEFAULTS;
    const struct gl_video_opts *opts = p->gl_opts_cache->opts;
    mp_csp_equalizer_state_get(p->video_eq, &cparams);
    pars->color_adjustment.brightness = cparams.brightness;
    pars->color_adjustment.contrast = cparams.contrast;
    pars->color_adjustment.hue = cparams.hue;
    pars->color_adjustment.saturation = cparams.saturation;
    pars->color_adjustment.gamma = cparams.gamma * opts->gamma;
}

static void draw_frame(struct priv *p, struct vo_frame *frame,
                       pl_tex target_tex, bool flip)
{
    pl_options pars = p->pars;
    const struct gl_video_opts *opts = p->gl_opts_cache->opts;

    struct pl_frame target = {
        .num_planes = 1,
        .planes[0] = {
            .texture = target_tex,
            .flipped = flip,
            .components = 4,
            .component_mapping = { 0, 1, 2, 3 },
        },
        .color = pl_color_space_srgb,
        .repr = pl_color_repr_rgb,
    };
    apply_crop(&target, p->dst, target_tex->params.w, target_tex->params.h);

    bool will_redraw = frame && frame->display_synced && frame->num_vsyncs > 1;
    bool can_interpolate = opts->interpolation && frame &&
                           frame->display_synced && !frame->still &&
                           frame->num_frames > 1;
    bool interpolate = can_interpolate &&
                       frame->ideal_frame_vsync_duration > 0;
    double pts_offset = interpolate ? frame->ideal_frame_vsync : 0;
    bool cache_frame = frame && (will_redraw || frame->still);

    // MPV_RENDER_PARAM_FLIP_Y already flipped the target; a stream-level
    // vflip cancels it.
    if (frame && frame->current && frame->current->params.vflip)
        target.planes[0].flipped = !target.planes[0].flipped;

    // Requesting a non-monotonic PTS is invalid for pl_queue; reset it to
    // recover. Clamp the check to 0, we don't have the previous frame anyway.
    struct pl_source_frame vpts;
    if (frame && frame->current && !p->want_reset) {
        if (pl_queue_peek(p->queue, 0, &vpts) &&
            frame->current->pts + MPMAX(0, pts_offset) < vpts.pts)
        {
            MP_VERBOSE(p, "Forcing queue refill, PTS(%f + %f | %f) < VPTS(%f)\n",
                       frame->current->pts, pts_offset,
                       frame->ideal_frame_vsync_duration, vpts.pts);
            p->want_reset = true;
        }
    }

    // Push all incoming frames into the frame queue
    if (frame) {
        if (p->want_reset) {
            // The queue requires monotonically increasing PTS, and the
            // renderer cache may still reference dropped frames.
            pl_queue_reset(p->queue);
            p->last_pts = 0.0;
            p->last_id = 0;
            p->want_reset = false;
            p->flush_cache = true;
        }

        if (p->flush_cache) {
            pl_renderer_flush_cache(p->rr);
            p->flush_cache = false;
        }

        for (int n = 0; n < frame->num_frames; n++) {
            uint64_t id = frame->frame_id + n;
            if (id <= p->last_id)
                continue; // ignore already seen frames

            struct mp_image *mpi = mp_image_new_ref(frame->frames[n]);
            struct frame_priv *fp = talloc_zero(mpi, struct frame_priv);
            fp->p = p;
            fp->signature = id;
            mpi->priv = fp;

            pl_queue_push(p->queue, &(struct pl_source_frame) {
                .pts = mpi->pts,
                .duration = interpolate ? frame->approx_duration : 0,
                .frame_data = mpi,
                .map = map_frame,
                .unmap = unmap_frame,
                .discard = discard_frame,
            });

            p->last_id = id;
        }
    }

    struct pl_frame_mix mix = {0};

    if (frame && frame->current) {
        struct pl_queue_params qparams = *pl_queue_params(
            .pts = frame->current->pts + pts_offset,
            .radius = pl_frame_mix_radius(&pars->params),
            .vsync_duration = interpolate ? frame->ideal_frame_vsync_duration : 0,
            .interpolation_threshold = opts->interpolation_threshold,
            .drift_compensation = 0,
        );

        // If the computed PTS precedes the first available frame (e.g. after
        // a reset), start from the first available frame instead.
        struct pl_source_frame first;
        if (pl_queue_peek(p->queue, 0, &first) && qparams.pts < first.pts) {
            if (first.pts != frame->current->pts)
                MP_VERBOSE(p, "Current PTS(%f) != VPTS(%f)\n",
                           frame->current->pts, first.pts);
            MP_VERBOSE(p, "Clamping first frame PTS from %f to %f\n",
                       qparams.pts, first.pts);
            qparams.pts = first.pts;
        }
        p->last_pts = qparams.pts;

        switch (pl_queue_update(p->queue, &mix, &qparams)) {
        case PL_QUEUE_ERR:
            MP_ERR(p, "Failed updating frames!\n");
            return;
        case PL_QUEUE_EOF:
            abort(); // we never signal EOF
        case PL_QUEUE_MORE:
            if (!frame->still)
                MP_DBG(p, "Render queue underrun.\n");
            break;
        case PL_QUEUE_OK:
            break;
        }
    } else {
        struct pl_queue_params qparams = *pl_queue_params(
            .pts = p->last_pts,
            .drift_compensation = 0,
        );
        pl_queue_update(p->queue, &mix, &qparams);
    }

    // A redraw invalidates the OSD but not the video frames.
    if (frame && frame->redraw)
        p->osd_sync++;

    // Must happen before pl_render_image_mix(), which otherwise derives the
    // crop from the (not yet acquired) hwdec plane texture.
    //
    // Frame IDs are used for signatures because mp_image pointers get
    // recycled; the OSD sync value keeps identical frames with a different
    // OSD distinct.
    uint64_t signatures[32];
    int num_sig = MPMIN(mix.num_frames, MP_ARRAY_SIZE(signatures));

    for (int i = 0; i < num_sig; i++) {
        struct pl_frame *image = (struct pl_frame *) mix.frames[i];
        struct mp_image *mpi = image->user_data;
        struct frame_priv *fp = mpi->priv;
        apply_crop(image, p->src, mpi->params.w, mpi->params.h);

        if (!opts->blend_subs) {
            image->num_overlays = 0;
            fp->osd_sync = 0;
        } else if (fp->osd_sync < p->osd_sync) {
            update_overlays(p, p->osd_res, OSD_DRAW_SUB_ONLY,
                            PL_OVERLAY_COORDS_DST_CROP, &fp->subs, image, mpi);
            fp->osd_sync = p->osd_sync;
        }

        // Fold the OSD sync value in to disambiguate identical frames with a
        // modified OSD.
        signatures[i] = fp->signature ^ (fp->osd_sync << 48);
    }
    if (num_sig) {
        mix.signatures = signatures;
        mix.num_frames = num_sig;
    }

    struct mp_image *ref = mix.num_frames > 0 ? mix.frames[0]->user_data : NULL;
    update_overlays(p, p->osd_res,
                    (frame && frame->current && opts->blend_subs)
                        ? OSD_DRAW_OSD_ONLY : 0,
                    PL_OVERLAY_COORDS_DST_FRAME, &p->osd_state, &target, ref);

    struct pl_render_params params = pars->params;
    params.preserve_mixing_cache = p->next_opts->inter_preserve &&
                                   !(frame && frame->still);
    if (frame && frame->still)
        params.frame_mixer = NULL;
    params.skip_caching_single_frame = !cache_frame;

    // The target is a client-provided texture, which cannot be flagged
    // blit_dst, so libplacebo must not clear it.
    params.background = PL_CLEAR_SKIP;
    params.border = PL_CLEAR_SKIP;

    if (!pl_render_image_mix(p->rr, &mix, &target, &params))
        MP_ERR(p, "Failed rendering frame!\n");
}

static bool check_format(struct render_backend *ctx, int imgfmt)
{
    struct priv *p = ctx->priv;

    if (ctx->hwdec_devs && ra_hwdec_get(&p->hwdec_ctx, imgfmt))
        return true;

    bool supported = format_supported(p, imgfmt, false);
    if (!supported)
        supported = format_supported(p, imgfmt, true);

    if (!supported && !ctx->hwdec_devs &&
        (mp_imgfmt_get_desc(imgfmt).flags & MP_IMGFLAG_HWACCEL))
    {
        MP_VERBOSE(p, "Format '%s' is a hwdec format, but the active API "
                   "backend provides no hwdec interop.\n",
                   mp_imgfmt_to_name(imgfmt));
    }

    return supported;
}

static void reconfig(struct render_backend *ctx, struct mp_image_params *params)
{
    struct priv *p = ctx->priv;

    p->want_reset = true;
}

static void reset(struct render_backend *ctx)
{
    struct priv *p = ctx->priv;

    p->want_reset = true;
}

static void update_external(struct render_backend *ctx, struct vo *vo)
{
    struct priv *p = ctx->priv;

    p->current_osd = vo ? vo->osd : NULL;
}

static void resize(struct render_backend *ctx, struct mp_rect *src,
                   struct mp_rect *dst, struct mp_osd_res *osd)
{
    struct priv *p = ctx->priv;

    if (src)
        p->src = *src;
    if (dst)
        p->dst = *dst;
    if (osd)
        p->osd_res = *osd;
}

static void video_screenshot(struct render_backend *ctx, struct vo_frame *frame,
                             struct voctrl_screenshot *args)
{
    struct priv *p = ctx->priv;
    const struct gl_video_opts *opts = p->gl_opts_cache->opts;
    struct mp_image *img = frame ? frame->current : NULL;
    struct pl_render_params params = p->pars->params;
    params.skip_caching_single_frame = true;
    params.preserve_mixing_cache = false;
    params.frame_mixer = NULL;

    if (!img)
        return;

    struct pl_frame source = {0};
    if (!upload_frame(p, &source, img)) {
        MP_ERR(p, "Failed uploading source image for screenshot.\n");
        return;
    }

    // Keep the screenshot independent of the system reference luminance.
    if (!pl_color_transfer_is_hdr(source.color.transfer))
        source.color.hdr.max_luma = 0;

    int w = img->w, h = img->h;
    if (img->params.rotate % 180 == 90)
        MPSWAP(int, w, h);

    apply_crop(&source, p->src, img->params.w, img->params.h);

    // Create target FBO, try high bit depth first
    int mpfmt = IMGFMT_RGBA;
    pl_tex fbo = NULL;
    for (int depth = opts->dither_depth > 8 ? 16 : 8; depth; depth -= 8) {
        mpfmt = depth == 16 ? IMGFMT_RGBA64 : IMGFMT_RGBA;
        pl_fmt fmt = pl_find_fmt(p->gpu, PL_FMT_UNORM, 4, depth, depth,
                                 PL_FMT_CAP_RENDERABLE | PL_FMT_CAP_HOST_READABLE);
        if (!fmt)
            continue;

        fbo = pl_tex_create(p->gpu, pl_tex_params(
            .w = w,
            .h = h,
            .format = fmt,
            .blit_dst = true,
            .renderable = true,
            .host_readable = true,
            .storable = fmt->caps & PL_FMT_CAP_STORABLE,
        ));
        if (fbo)
            break;
    }

    if (!fbo) {
        MP_ERR(p, "Failed creating target FBO for screenshot!\n");
        goto done;
    }

    struct pl_frame target = {
        .color = pl_color_space_srgb,
        .repr = pl_color_repr_rgb,
        .num_planes = 1,
        .planes[0] = {
            .texture = fbo,
            .components = 4,
            .component_mapping = { 0, 1, 2, 3 },
        },
    };

    // Round-trip back to sRGB if the source is also sRGB.
    if ((opts->treat_srgb_as_power22 & 1) &&
        target.color.transfer == PL_COLOR_TRC_SRGB &&
        img->params.color.transfer == PL_COLOR_TRC_SRGB)
    {
        target.color.transfer = PL_COLOR_TRC_GAMMA22;
    }

    apply_crop(&target, (struct mp_rect){ 0, 0, w, h }, fbo->params.w,
               fbo->params.h);

    struct mp_osd_res osd = {
        .w = w,
        .h = h,
        .display_par = 1.0, // Screenshots have square pixels
    };
    struct osd_state sub_state = {0};
    update_overlays(p, osd, 0, PL_OVERLAY_COORDS_DST_FRAME, &sub_state, &target,
                    img);

    if (!pl_render_image(p->rr, &source, &target, &params)) {
        MP_ERR(p, "Failed rendering frame!\n");
        goto done;
    }

    args->res = mp_image_alloc(mpfmt, w, h);
    if (!args->res) {
        MP_ERR(p, "Failed to allocate screenshot image.\n");
        goto done;
    }

    args->res->params.color.primaries = target.color.primaries;
    args->res->params.color.transfer = target.color.transfer;
    args->res->params.color.hdr = target.color.hdr;
    args->res->params.repr.levels = target.repr.levels;

    bool ok = pl_tex_download(p->gpu, pl_tex_transfer_params(
        .tex = fbo,
        .ptr = args->res->planes[0],
        .row_pitch = args->res->stride[0],
    ));

    if (!ok) {
        MP_ERR(p, "Screenshot texture download failed.\n");
        TA_FREEP(&args->res);
    }

done:
    pl_tex_destroy(p->gpu, &fbo);
    recycle_subs(p, &sub_state);
    talloc_free(sub_state.overlays);
    upload_frame_cleanup(p, &source);
}

static int set_parameter(struct render_backend *ctx, mpv_render_param param)
{
    return MPV_ERROR_NOT_IMPLEMENTED;
}

static int get_target_size(struct render_backend *ctx, mpv_render_param *params,
                           int *out_w, int *out_h)
{
    struct priv *p = ctx->priv;

    pl_tex tex;
    int err = p->context->fns->wrap_fbo(p->context, params, &tex);
    if (err < 0)
        return err;
    *out_w = tex->params.w;
    *out_h = tex->params.h;
    return 0;
}

static int render(struct render_backend *ctx, mpv_render_param *params,
                  struct vo_frame *frame)
{
    struct priv *p = ctx->priv;

    pl_tex tex;
    int err = p->context->fns->wrap_fbo(p->context, params, &tex);
    if (err < 0)
        return err;

    update_options(ctx);

    bool flip = GET_MPV_RENDER_PARAM(params, MPV_RENDER_PARAM_FLIP_Y, int, 0);
    draw_frame(p, frame, tex, flip);

    if (p->context->fns->done_frame)
        p->context->fns->done_frame(p->context);
    return 0;
}

static void screenshot(struct render_backend *ctx, struct vo_frame *frame,
                       struct voctrl_screenshot *args)
{
    args->res = NULL;
    video_screenshot(ctx, frame, args);
}

static void load_hwdec_api(void *ctx, struct hwdec_imgfmt_request *params)
{
    struct render_backend *rb = ctx;
    struct priv *p = rb->priv;

    if (rb->hwdec_devs)
        ra_hwdec_ctx_load_fmt(&p->hwdec_ctx, rb->hwdec_devs, params);
}

static int init(struct render_backend *ctx, mpv_render_param *params)
{
    ctx->priv = talloc_zero(ctx, struct priv);
    struct priv *p = ctx->priv;

    p->log = ctx->log;
    p->global = ctx->global;

    char *api = get_mpv_render_param(params, MPV_RENDER_PARAM_API_TYPE, NULL);
    if (!api)
        return MPV_ERROR_INVALID_PARAMETER;

    const struct libmpv_gpu_next_context_fns *backend = NULL;
    for (int n = 0; context_backends[n]; n++) {
        if (strcmp(context_backends[n]->api_name, api) == 0)
            backend = context_backends[n];
    }
    if (!backend)
        return MPV_ERROR_NOT_IMPLEMENTED;

    p->context = talloc_zero(p, struct libmpv_gpu_next_context);
    *p->context = (struct libmpv_gpu_next_context) {
        .log = ctx->log,
        .global = ctx->global,
        .fns = backend,
    };

    int err = backend->init(p->context, params);
    if (err < 0) {
        // The caller still invokes destroy() after a failed init(), so tear
        // down the partially initialized API context here and forget it.
        p->context->fns->destroy(p->context);
        TA_FREEP(&p->context);
        return err;
    }

    // For the time being
    p->pllog = p->context->pl_log;
    p->gpu = p->context->gpu;
    p->rr = p->context->rr;
    p->opts_cache = m_config_cache_alloc(p, ctx->global, &gl_next_conf);
    p->next_opts = p->opts_cache->opts;

    p->gl_opts_cache = m_config_cache_alloc(p, ctx->global, &gl_video_conf);

    p->video_eq = mp_csp_equalizer_create(p, ctx->global);
    mp_mutex_init(&p->dr_lock);

    p->queue = pl_queue_create(p->gpu);
    p->osd_fmt[SUBBITMAP_LIBASS] = pl_find_named_fmt(p->gpu, "r8");
    p->osd_fmt[SUBBITMAP_BGRA] = pl_find_named_fmt(p->gpu, "bgra8");
    p->osd_sync = 1;

    p->pars = pl_options_alloc(p->pllog);
    if (!p->pars) {
        MP_ERR(p, "Failed allocating libplacebo render options!\n");
        return MPV_ERROR_VO_INIT_FAILED;
    }

    ctx->hwdec_devs = hwdec_devices_create();
    if (!ctx->hwdec_devs)
        return MPV_ERROR_VO_INIT_FAILED;

    if (p->context->ra_ctx) {
        struct gl_video_opts *gl_opts = p->gl_opts_cache->opts;
        hwdec_devices_set_loader(ctx->hwdec_devs, load_hwdec_api, ctx);

        p->hwdec_ctx = (struct ra_hwdec_ctx) {
            .log = p->log,
            .global = p->global,
            .ra_ctx = p->context->ra_ctx,
        };
        // Preload the interop drivers, otherwise lazy loading would force a
        // HW download since check_format() results are cached.
        ra_hwdec_ctx_init(&p->hwdec_ctx, ctx->hwdec_devs, gl_opts->hwdec_interop,
                          true);
    }

    ctx->driver_caps = VO_CAP_ROTATE90 | VO_CAP_VFLIP;
    return 0;
}

static void destroy(struct render_backend *ctx)
{
    struct priv *p = ctx->priv;

    // Partial init failure: nothing but the API context was created, and it
    // has already been torn down by init().
    if (!p || !p->gpu)
        return;

    // Drain any in-flight uploads.
    if (p->gpu)
        pl_gpu_finish(p->gpu);

    pl_queue_destroy(&p->queue); // destroy this first
    for (int i = 0; i < MP_ARRAY_SIZE(p->osd_state.entries); i++) {
        struct osd_entry *entry = &p->osd_state.entries[i];
        pl_tex_destroy(p->gpu, &entry->tex);
        talloc_free(entry->parts);
    }
    talloc_free(p->osd_state.overlays);
    for (int i = 0; i < p->num_sub_tex; i++)
        pl_tex_destroy(p->gpu, &p->sub_tex[i]);
    talloc_free(p->sub_tex);

    if (ctx->hwdec_devs) {
        slot_uninit(p, &p->hwdec);
        ra_hwdec_ctx_uninit(&p->hwdec_ctx);
        hwdec_devices_set_loader(ctx->hwdec_devs, NULL, NULL);
        hwdec_devices_destroy(ctx->hwdec_devs);
        ctx->hwdec_devs = NULL;
    }

    mp_mutex_lock(&p->dr_lock);
    for (int i = 0; i < p->num_dr_buffers; i++)
        pl_buf_destroy(p->gpu, &p->dr_buffers[i]);
    p->num_dr_buffers = 0;
    p->gpu = NULL;
    mp_mutex_unlock(&p->dr_lock);
    talloc_free(p->dr_buffers);
    mp_mutex_destroy(&p->dr_lock);

    if (p->pars)
        pl_options_free(&p->pars);

    if (p->context) {
        p->context->fns->destroy(p->context);
        talloc_free(p->context);
        p->context = NULL;
    }
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
    .destroy = destroy,
};
