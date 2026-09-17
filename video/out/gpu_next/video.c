#include <dirent.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#include <libplacebo/colorspace.h>
#include <libplacebo/gpu.h>
#include <libplacebo/options.h>
#include <libplacebo/renderer.h>
#include <libplacebo/utils/frame_queue.h>
#include <libplacebo/utils/upload.h>

#include "common/common.h"
#include "common/msg.h"
#include "misc/io_utils.h"
#include "misc/path_utils.h"
#include "options/m_config.h"
#include "options/path.h"
#include "stream/stream.h"
#include "sub/draw_bmp.h"
#include "sub/osd.h"
#include "ta/ta_talloc.h"
#include "video/csputils.h"
#include "video/hwdec.h"
#include "video/img_format.h"
#include "video/mp_image.h"
#include "video/out/gpu/hwdec.h"
#include "video/out/gpu/video.h"
#include "video/out/placebo/ra_pl.h"
#include "video/out/vo.h"
#include "video/out/gpu_next/vo_gpu_next.h"
#include "video.h"

#if HAVE_GL && defined(PL_HAVE_OPENGL)
#include <libplacebo/opengl.h>
#include "video/out/opengl/ra_gl.h"
#endif

#if HAVE_D3D11 && defined(PL_HAVE_D3D11)
#include <libplacebo/d3d11.h>
#include "video/out/d3d11/ra_d3d11.h"
#include "osdep/windows_utils.h"
#endif

// Upper bound on the number of host-mapped buffers handed out through
// pl_video_get_image() and still referenced by the client.
#define MP_MAX_DR_BUFFERS 128

struct osd_entry {
    pl_tex tex;
    struct pl_overlay_part *parts;
    int num_parts;
};

struct pl_osd_state {
    struct osd_entry entries[MAX_OSD_PARTS];
    struct pl_overlay overlays[MAX_OSD_PARTS];
};

// Persisted libplacebo shader (pass) cache, mirroring the "shader" cache in
// vo_gpu_next.c. Without it every process start has to recompile all GPU
// passes from scratch (~1 second for the D3D11 backend).
//
// `dir` is a child of the owning pl_video, not of this struct, which is
// embedded in pl_video and is therefore not a valid talloc pointer itself.
struct pl_shader_cache {
    struct mp_log *log;
    struct mpv_global *global;
    char *dir;                  // NULL if no cache directory is available
    const char *name;           // filename prefix, "shader"
    size_t size_limit;          // soft cache size limit, in bytes
};

struct pl_video {
    struct mp_log *log;
    struct mpv_global *global;
    pl_gpu gpu;                 // borrowed from the API context layer
    pl_renderer renderer;       // owned by the API context layer
    pl_log pllog;               // borrowed from the API context layer
    pl_queue queue;

    uint64_t last_id;           // Highest frame_id pushed so far (reset-aware).
    double last_pts;
    bool want_reset;            // Queue must be flushed before the next push.
    uint64_t osd_sync;          // Bumped whenever the OSD changes.

    struct mp_rect current_dst;
    struct mp_rect current_src;
    struct osd_state *current_osd_state;

    struct mp_osd_res osd_res;
    struct pl_osd_state osd_state;
    pl_fmt osd_fmt[SUBBITMAP_COUNT];
    pl_tex *sub_tex;
    int num_sub_tex;

    // Hardware decoding interop. Optional; only set up if the backend provides
    // an ra_ctx (currently D3D11). The mapper is reused across frames and
    // reconfigured when the source parameters change.
    struct mp_hwdec_devices *hwdec_devs;
    struct ra_hwdec_ctx hwdec_ctx;
    struct ra_hwdec_mapper *hwdec_mapper;

    struct mp_csp_equalizer_state *video_eq;

    // Direct rendering buffers handed out via pl_video_get_image(). They are
    // host-mapped, so upload_frame() can reference them without a copy.
    mp_mutex dr_lock;
    pl_buf *dr_buffers;
    int num_dr_buffers;

    struct m_config_cache *opts_cache;
    struct gl_next_opts *opts;

    // Render options mapped from gl_video_opts + gl_next_opts, kept persistent
    // so libplacebo can recompile only what actually changed between frames.
    struct m_config_cache *gl_opts_cache;
    struct gl_video_opts *gl_opts;
    pl_options pars;

    // Persistent shader cache, also registered on the shared pl_gpu.
    struct pl_shader_cache shader_cache;
    pl_cache pl_cache;
};

struct frame_priv {
    struct pl_video *p;
    struct ra_hwdec *hwdec;
    uint64_t signature;         // unique per queue entry, see render()
    uint64_t osd_sync;          // OSD version baked into this frame's overlays
};

static pl_buf get_dr_buf(struct pl_video *p, const uint8_t *ptr)
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

        // If the frame points into one of our own host-mapped buffers, upload
        // it by reference instead of copying the pixel data.
        pl_buf buf = get_dr_buf(p, data[n].pixels);
        if (buf) {
            data[n].buf = buf;
            data[n].buf_offset = (uint8_t *)data[n].pixels - buf->data;
            data[n].pixels = NULL;
        }

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

static bool hwdec_reconfig(struct pl_video *p, struct ra_hwdec *hwdec,
                           const struct mp_image_params *par)
{
    if (p->hwdec_mapper) {
        if (mp_image_params_static_equal(par, &p->hwdec_mapper->src_params)) {
            p->hwdec_mapper->src_params.repr.dovi = par->repr.dovi;
            p->hwdec_mapper->dst_params.repr.dovi = par->repr.dovi;
            p->hwdec_mapper->src_params.color.hdr = par->color.hdr;
            p->hwdec_mapper->dst_params.color.hdr = par->color.hdr;
            return p->hwdec_mapper;
        } else {
            ra_hwdec_mapper_free(&p->hwdec_mapper);
        }
    }

    p->hwdec_mapper = ra_hwdec_mapper_create(hwdec, par);
    if (!p->hwdec_mapper) {
        MP_ERR(p, "Initializing texture for hardware decoding failed.\n");
        return NULL;
    }

    return p->hwdec_mapper;
}

// For RAs not based on ra_pl, this creates a new pl_tex wrapper
static pl_tex hwdec_get_tex(struct pl_video *p, int n)
{
    struct ra_tex *ratex = p->hwdec_mapper->tex[n];
    struct ra *ra = p->hwdec_mapper->ra;
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
    return false;
}

static bool hwdec_acquire(pl_gpu gpu, struct pl_frame *frame)
{
    struct mp_image *mpi = frame->user_data;
    struct frame_priv *fp = mpi->priv;
    struct pl_video *p = fp->p;

    // Map lazily (as vo_gpu_next does): libplacebo may hold several queued
    // frames at once, and a single ra_hwdec_mapper can only map one surface.
    if (!hwdec_reconfig(p, fp->hwdec, &mpi->params))
        return false;

    if (ra_hwdec_mapper_map(p->hwdec_mapper, mpi) < 0) {
        MP_ERR(p, "Mapping hardware decoded surface failed.\n");
        return false;
    }

    for (int n = 0; n < frame->num_planes; n++) {
        frame->planes[n].texture = hwdec_get_tex(p, n);
        if (!frame->planes[n].texture) {
            // Roll back the planes mapped so far; release_frame() is only
            // invoked once acquire() reported success.
            for (int i = 0; i < n; i++)
                pl_tex_destroy(p->gpu, &frame->planes[i].texture);
            ra_hwdec_mapper_unmap(p->hwdec_mapper);
            return false;
        }
    }

    return true;
}

static void hwdec_release(pl_gpu gpu, struct pl_frame *frame)
{
    struct mp_image *mpi = frame->user_data;
    struct frame_priv *fp = mpi->priv;
    struct pl_video *p = fp->p;

    // For RAs not based on ra_pl these are wrappers created by hwdec_get_tex();
    // for ra_pl they are owned by the mapper and must not be destroyed here.
    if (!ra_pl_get(p->hwdec_mapper->ra)) {
        for (int n = 0; n < frame->num_planes; n++)
            pl_tex_destroy(p->gpu, &frame->planes[n].texture);
    }
    ra_hwdec_mapper_unmap(p->hwdec_mapper);
}

static bool map_frame(pl_gpu gpu, pl_tex *tex, const struct pl_source_frame *src,
                      struct pl_frame *frame)
{
    struct mp_image *mpi = src->frame_data;
    struct frame_priv *fp = mpi->priv;
    struct pl_video *p = fp->p;

    struct mp_image_params par = mpi->params;
    fp->hwdec = p->hwdec_devs ? ra_hwdec_get(&p->hwdec_ctx, mpi->imgfmt) : NULL;
    if (fp->hwdec) {
        // Reconfig the mapper here (potentially creating it) to access
        // `dst_params`.
        if (!hwdec_reconfig(p, fp->hwdec, &mpi->params))
            return false;

        par = p->hwdec_mapper->dst_params;
    }

    mp_image_params_guess_csp(&par);

    *frame = (struct pl_frame) {
        .color = par.color,
        .repr = par.repr,
        .rotation = par.rotate / 90,
        .user_data = mpi,
    };

    if (fp->hwdec) {
        struct mp_imgfmt_desc desc = mp_imgfmt_get_desc(par.imgfmt);
        frame->acquire = hwdec_acquire;
        frame->release = hwdec_release;
        frame->num_planes = desc.num_planes;
        for (int n = 0; n < frame->num_planes; n++) {
            struct pl_plane *plane = &frame->planes[n];
            int *map = plane->component_mapping;
            for (int c = 0; c < mp_imgfmt_desc_get_num_comps(&desc); c++) {
                if (desc.comps[c].plane != n)
                    continue;

                // Sort by component offset
                uint8_t offset = desc.comps[c].offset;
                int index = plane->components++;
                while (index > 0 && desc.comps[map[index - 1]].offset > offset) {
                    map[index] = map[index - 1];
                    index--;
                }
                map[index] = c;
            }
        }

        pl_frame_set_chroma_location(frame, par.chroma_location);
        return true;
    }

    if (!upload_frame(p, frame, mpi))
        return false;

    frame->user_data = mpi;
    return true;
}

static void unmap_frame(pl_gpu gpu, struct pl_frame *frame,
                        const struct pl_source_frame *src)
{
    struct mp_image *mpi = src->frame_data;
    struct frame_priv *fp = mpi->priv;
    struct pl_video *p = fp->p;

    // hwdec frames are mapped by hwdec_acquire() and torn down by
    // hwdec_release(), which libplacebo pairs with a successful acquire().
    if (!fp->hwdec)
        upload_frame_cleanup(p, frame);
    talloc_free(mpi);
}

static void discard_frame(const struct pl_source_frame *src)
{
    // Only called for frames that were never mapped, so there is nothing to
    // release. map_frame() rolls back its own partial mappings on failure.
    talloc_free(src->frame_data);
}

static char *cache_filepath(void *ta_ctx, char *dir, const char *prefix,
                            uint64_t key)
{
    bstr filename = {0};
    bstr_xappend_asprintf(ta_ctx, &filename, "%s_%016" PRIx64, prefix, key);
    return mp_path_join_bstr(ta_ctx, bstr0(dir), filename);
}

static pl_cache_obj cache_load_obj(void *p, uint64_t key)
{
    struct pl_shader_cache *c = p;
    void *ta_ctx = talloc_new(NULL);
    pl_cache_obj obj = {0};

    if (!c->dir)
        goto done;

    char *filepath = cache_filepath(ta_ctx, c->dir, c->name, key);
    if (!filepath)
        goto done;

    if (stat(filepath, &(struct stat){0}))
        goto done;

    int64_t load_start = mp_time_ns();
    struct bstr data = stream_read_file(filepath, ta_ctx, c->global,
                                        STREAM_MAX_READ_SIZE);
    int64_t load_end = mp_time_ns();
    MP_DBG(c, "%s: key(%" PRIx64 "), size(%zu), load time(%.3f ms)\n",
           __func__, key, data.len,
           MP_TIME_NS_TO_MS(load_end - load_start));

    obj = (pl_cache_obj){
        .key = key,
        .data = talloc_steal(NULL, data.start),
        .size = data.len,
        .free = talloc_free,
    };

done:
    talloc_free(ta_ctx);
    return obj;
}

static void cache_save_obj(void *p, pl_cache_obj obj)
{
    const struct pl_shader_cache *c = p;
    void *ta_ctx = talloc_new(NULL);

    if (!c->dir)
        goto done;

    char *filepath = cache_filepath(ta_ctx, c->dir, c->name, obj.key);
    if (!filepath)
        goto done;

    if (!obj.data || !obj.size) {
        unlink(filepath);
        goto done;
    }

    // Don't save if already exists.
    struct stat st;
    if (!stat(filepath, &st) && st.st_size == obj.size) {
        MP_DBG(c, "%s: key(%" PRIx64 "), size(%zu)\n", __func__, obj.key, obj.size);
        goto done;
    }

    int64_t save_start = mp_time_ns();
    mp_save_to_file(filepath, obj.data, obj.size);
    int64_t save_end = mp_time_ns();
    MP_DBG(c, "%s: key(%" PRIx64 "), size(%zu), save time(%.3f ms)\n",
           __func__, obj.key, obj.size,
           MP_TIME_NS_TO_MS(save_end - save_start));

done:
    talloc_free(ta_ctx);
}

// Set up the shader cache and register it on the GPU.
//
// Called from pl_video_init() first, but the user config paths may not be ready
// yet: for libmpv the render context can be created before mp_initialize() runs
// mp_init_paths(). In that case the call is retried lazily from the first
// render, at which point the client is certainly initialized.
static void shader_cache_init(struct pl_video *p)
{
    if (p->pl_cache || !p->gl_opts->shader_cache)
        return; // already initialized, or caching disabled

    struct pl_shader_cache *c = &p->shader_cache;
    c->log = p->log;
    c->global = p->global;
    c->name = "shader";
    c->size_limit = 128 << 20;

    if (!c->dir) {
        // Retry every frame until the client's config paths are available.
        // Use a temporary allocation so the repeated attempts do not leak a
        // talloc string per frame.
        const char *dir_opt = p->gl_opts->shader_cache_dir;
        void *tmp = talloc_new(NULL);
        char *dir;
        if (dir_opt && dir_opt[0]) {
            dir = mp_get_user_path(tmp, p->global, dir_opt);
        } else {
            dir = mp_find_user_file(tmp, p->global, "cache", "");
        }
        if (!dir || !dir[0]) {
            talloc_free(tmp);
            return;
        }
        c->dir = talloc_steal(p, dir);
        talloc_free(tmp);
        mp_mkdirp(c->dir);
    }

    p->pl_cache = pl_cache_create(pl_cache_params(
        .log = p->pllog,
        .get = cache_load_obj,
        .set = cache_save_obj,
        .priv = c
    ));
    if (p->pl_cache)
        pl_gpu_set_cache(p->gpu, p->pl_cache);
}

struct file_entry {
    char *filepath;
    size_t size;
    time_t atime;
};

static int compare_atime(const void *a, const void *b)
{
    return (((struct file_entry *)b)->atime - ((struct file_entry *)a)->atime);
}

// Unregister the cache from the GPU and prune old entries past the soft limit.
static void cache_uninit(struct pl_video *p, struct pl_shader_cache *cache)
{
    if (!p->pl_cache)
        return;

    // The pl_gpu may outlive us (the API context owns it).
    pl_gpu_set_cache(p->gpu, NULL);
    pl_cache_destroy(&p->pl_cache);

    if (!cache->dir)
        return;

    void *ta_ctx = talloc_new(NULL);
    struct file_entry *files = NULL;
    size_t num_files = 0;

    DIR *d = opendir(cache->dir);
    if (!d)
        goto done;

    struct dirent *dir;
    while ((dir = readdir(d)) != NULL) {
        char *filepath = mp_path_join(ta_ctx, cache->dir, dir->d_name);
        if (!filepath)
            continue;
        struct stat filestat;
        if (stat(filepath, &filestat))
            continue;
        if (!S_ISREG(filestat.st_mode))
            continue;
        bstr fname = bstr0(dir->d_name);
        if (!bstr_eatstart0(&fname, cache->name))
            continue;
        if (!bstr_eatstart0(&fname, "_"))
            continue;
        if (fname.len != 16) // %016x
            continue;
        MP_TARRAY_APPEND(ta_ctx, files, num_files,
                         (struct file_entry){
                             .filepath = filepath,
                             .size     = filestat.st_size,
                             .atime    = filestat.st_atime,
                         });
    }
    closedir(d);

    if (!num_files)
        goto done;

    qsort(files, num_files, sizeof(struct file_entry), compare_atime);

    time_t t = time(NULL);
    size_t cache_size = 0;
    size_t cache_limit = cache->size_limit ? cache->size_limit : SIZE_MAX;
    for (int i = 0; i < num_files; i++) {
        // Remove files that exceed the size limit but are older than one day.
        // This allows for temporarily maintaining a larger cache size while
        // adjusting the configuration; unused entries are cleared the next day.
        cache_size += files[i].size;
        double rel_use = difftime(t, files[i].atime);
        if (cache_size > cache_limit && rel_use > 60 * 60 * 24) {
            MP_VERBOSE(p, "Removing %s | size: %9zu bytes | last used: %9d seconds ago\n",
                       files[i].filepath, files[i].size, (int)rel_use);
            unlink(files[i].filepath);
        }
    }

done:
    talloc_free(ta_ctx);
}

struct pl_video *pl_video_init(struct mpv_global *global, struct mp_log *log,
                               pl_log pllog, pl_gpu gpu, pl_renderer renderer)
{
    struct pl_video *p = talloc_zero(NULL, struct pl_video);
    p->log = log;
    p->global = global;
    p->gpu = gpu;
    p->renderer = renderer;
    p->pllog = pllog;
    p->queue = pl_queue_create(gpu);

    // Pre-find the texture formats we'll need for OSD bitmaps for efficiency.
    p->osd_fmt[SUBBITMAP_LIBASS] = pl_find_fmt(gpu, PL_FMT_UNORM, 1, 8, 8, 0);
    p->osd_fmt[SUBBITMAP_BGRA]   = pl_find_fmt(gpu, PL_FMT_UNORM, 4, 8, 8, 0);

    p->video_eq = mp_csp_equalizer_create(p, global);

    mp_mutex_init(&p->dr_lock);

    p->opts_cache = m_config_cache_alloc(p, global, &gl_next_conf);
    p->opts = p->opts_cache->opts;

    // User-facing scaler/deband/tone-mapping/... options, like vo_gpu_next.
    p->gl_opts_cache = m_config_cache_alloc(p, global, &gl_video_conf);
    p->gl_opts = p->gl_opts_cache->opts;

    // Persistent render options; libplacebo keys its pass/shader caches off
    // these, so they must be stable across frames.
    p->pars = pl_options_alloc(p->pllog);
    if (!p->pars) {
        MP_ERR(p, "Failed allocating libplacebo render options!\n");
        talloc_free(p);
        return NULL;
    }

    // Install a persistent shader cache before anything renders, so that the
    // first frame does not have to compile every pass from scratch. This may
    // silently fail if the user config paths are not ready yet; pl_video_render
    // retries it once per frame until it succeeds.
    shader_cache_init(p);

    return p;
}

void pl_video_init_hwdecs(struct pl_video *p, struct ra_ctx *ra_ctx,
                          struct mp_hwdec_devices *devs, const char *interop)
{
    p->hwdec_devs = devs;
    p->hwdec_ctx = (struct ra_hwdec_ctx) {
        .log = p->log,
        .global = p->global,
        .ra_ctx = ra_ctx,
    };

    // Preload the interop drivers: the render backend caches check_format()
    // results (imgfmt_supported[] in vo_libmpv.c) before the decoder can
    // request a format, so lazy loading would report hwdec formats as
    // unsupported and force a HW download. The 'gpu' backend does the same via
    // gl_video_init_hwdecs(..., true).
    ra_hwdec_ctx_init(&p->hwdec_ctx, devs, interop, true);
}

void pl_video_load_hwdecs(struct pl_video *p, struct hwdec_imgfmt_request *params)
{
    ra_hwdec_ctx_load_fmt(&p->hwdec_ctx, p->hwdec_devs, params);
}

static void free_dr_buf(void *opaque, uint8_t *data)
{
    struct pl_video *p = opaque;

    mp_mutex_lock(&p->dr_lock);

    // Set to NULL by pl_video_uninit(); the GPU may already be gone by the
    // time outstanding frames are released.
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
}

struct mp_image *pl_video_get_image(struct pl_video *p, int imgfmt, int w, int h,
                                    int stride_align, int flags)
{
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

    // Pad by one row so that get_dr_buf()'s range check still covers the last
    // plane, whose end may be one past the size computed above.
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
    // Refuse to grow the pool without bound; exceeding the cap means the
    // client is not recycling the frames it was given.
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

bool pl_video_check_format(struct pl_video *p, int imgfmt)
{
    // hwdec formats are mapped by the interop, not uploaded to libplacebo.
    if (p->hwdec_devs && ra_hwdec_get(&p->hwdec_ctx, imgfmt))
        return true;

    struct pl_bit_encoding bits;
    struct pl_plane_data data[4] = {0};
    int planes = plane_data_from_imgfmt(data, &bits, imgfmt, false);
    if (!planes)
        planes = plane_data_from_imgfmt(data, &bits, imgfmt, true);
    if (!planes) {
        if (!p->hwdec_devs && mp_imgfmt_get_desc(imgfmt).flags & MP_IMGFLAG_HWACCEL)
            MP_VERBOSE(p, "Format '%s' is a hwdec format, but the active API "
                       "backend provides no hwdec interop.\n",
                       mp_imgfmt_to_name(imgfmt));
        return false;
    }

    for (int i = 0; i < planes; i++) {
        if (!pl_plane_find_fmt(p->gpu, NULL, &data[i]))
            return false;
    }

    return true;
}

void pl_video_uninit(struct pl_video **p_ptr)
{
    struct pl_video *p = *p_ptr;
    if (!p)
        return;

    ra_hwdec_mapper_free(&p->hwdec_mapper);
    if (p->hwdec_devs)
        ra_hwdec_ctx_uninit(&p->hwdec_ctx);

    // The queue holds mapped frames whose unmap/discard callbacks may still
    // reference our own DR buffers, so drain it first to release them.
    pl_queue_destroy(&p->queue);

    for (int i = 0; i < MAX_OSD_PARTS; i++) {
        struct osd_entry *entry = &p->osd_state.entries[i];
        pl_tex_destroy(p->gpu, &entry->tex);
        talloc_free(entry->parts);
    }
    for (int i = 0; i < p->num_sub_tex; i++)
        pl_tex_destroy(p->gpu, &p->sub_tex[i]);
    talloc_free(p->sub_tex);

    if (p->pars)
        pl_options_free(&p->pars);

    // Must happen while the GPU is still alive, since the cache is on it.
    cache_uninit(p, &p->shader_cache);

    // Stop free_dr_buf() from touching the GPU: it may be invoked after the
    // device is torn down by the API context layer, which owns it.
    mp_mutex_lock(&p->dr_lock);
    for (int i = 0; i < p->num_dr_buffers; i++)
        pl_buf_destroy(p->gpu, &p->dr_buffers[i]);
    p->num_dr_buffers = 0;
    p->gpu = NULL;
    mp_mutex_unlock(&p->dr_lock);
    talloc_free(p->dr_buffers);
    mp_mutex_destroy(&p->dr_lock);

    // Free the cache directory string, a child of `p` rather than of the
    // (embedded) shader_cache struct.
    talloc_free(p->shader_cache.dir);
    p->shader_cache.dir = NULL;

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
        // Grow the texture as needed, but shrink it again when the new bitmap
        // is substantially smaller, so that a single large subtitle or OSD
        // element does not pin a large texture in the pool forever.
        int cur_w = entry->tex ? entry->tex->params.w : 0;
        int cur_h = entry->tex ? entry->tex->params.h : 0;
        bool shrink = item->packed_w * 2 <= cur_w && item->packed_h * 2 <= cur_h;
        bool ok = pl_tex_recreate(p->gpu, &entry->tex, &(struct pl_tex_params) {
            .format = tex_fmt,
            .w = shrink ? item->packed_w : MPMAX(item->packed_w, cur_w),
            .h = shrink ? item->packed_h : MPMAX(item->packed_h, cur_h),
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

// Translate mpv's rotated/flipped display rect into libplacebo's unrotated
// `pl_frame.crop` convention, mirroring apply_crop() in vo_gpu_next.c.
static void apply_crop(struct pl_frame *frame, struct mp_rect crop,
                       int width, int height)
{
    frame->crop = (struct pl_rect2df) {
        .x0 = crop.x0,
        .y0 = crop.y0,
        .x1 = crop.x1,
        .y1 = crop.y1,
    };

    // mpv gives us rotated/flipped rects, libplacebo expects unrotated
    pl_rect2df_rotate(&frame->crop, -frame->rotation);
    if (frame->crop.x1 < frame->crop.x0) {
        frame->crop.x0 = width - frame->crop.x0;
        frame->crop.x1 = width - frame->crop.x1;
    }

    if (frame->crop.y1 < frame->crop.y0) {
        frame->crop.y0 = height - frame->crop.y0;
        frame->crop.y1 = height - frame->crop.y1;
    }
}

void pl_video_render(struct pl_video *p, struct vo_frame *frame, pl_tex target_tex)
{
    // Refresh the option caches before anything reads them.
    m_config_cache_update(p->gl_opts_cache);
    m_config_cache_update(p->opts_cache);

    // May become possible only after the client has set up its config paths.
    shader_cache_init(p);

    struct pl_frame target = {
        .num_planes = 1,
        .planes[0] = { .texture = target_tex, .components = 4, .component_mapping = {0,1,2,3} },
        .color = pl_color_space_srgb,
        .repr = pl_color_repr_rgb,
    };
    apply_crop(&target, p->current_dst, target_tex->params.w,
               target_tex->params.h);

    // Mirror vo_gpu_next's draw_frame() decisions. Interpolation is only
    // meaningful when it is enabled, the client drives us at a fixed vsync
    // rate, is not redrawing a still frame and provides future frames.
    bool will_redraw = frame && frame->display_synced && frame->num_vsyncs > 1;
    bool can_interpolate = p->gl_opts->interpolation && frame &&
                           frame->display_synced && !frame->still &&
                           frame->num_frames > 1;
    bool interpolate = can_interpolate &&
                       frame->ideal_frame_vsync_duration > 0;
    double pts_offset = interpolate ? frame->ideal_frame_vsync : 0;

    // If the current frame's PTS is behind the first queued frame (which can
    // happen right after a seek or when the vsync offset moves us backwards),
    // pl_queue would reject the non-monotonic PTS. Force a refill instead of
    // letting libplacebo abort or silently drop frames.
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

    // The frame is delivered as a list of future frames, frames[0] being the
    // current one. Push every not-yet-seen frame so that libplacebo can
    // interpolate and mix between them.
    if (frame) {
        for (int n = 0; n < frame->num_frames; n++) {
            uint64_t id = frame->frame_id + n;

            if (p->want_reset) {
                // Discard all queued frames and start over. The PTS counter
                // must be reset as well, because pl_queue validates that the
                // PTS passed to pl_queue_update() is monotonically increasing.
                pl_renderer_flush_cache(p->renderer);
                pl_queue_reset(p->queue);
                p->last_pts = 0.0;
                p->last_id = 0;
                p->want_reset = false;
            }

            if (id <= p->last_id)
                continue; // ignore already seen frames

            struct mp_image *mpi = mp_image_new_ref(frame->frames[n]);
            struct frame_priv *fp = talloc_zero(mpi, struct frame_priv);
            fp->p = p;
            fp->signature = id;
            mpi->priv = fp;

            pl_queue_push(p->queue, &(struct pl_source_frame) {
                .pts = mpi->pts,
                // Zero duration disables frame mixing for this entry; only
                // supply it when interpolation is actually possible, like
                // vo_gpu_next does.
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
            .radius = pl_frame_mix_radius(&p->pars->params),
            // Only interpolate when the display is synchronised and the client
            // gave us a usable vsync duration. Otherwise 0 disables the
            // frame-mixing interpolation entirely.
            .vsync_duration = interpolate ? frame->ideal_frame_vsync_duration : 0,
            .interpolation_threshold = p->gl_opts->interpolation_threshold,
        );
#if PL_API_VER >= 340
        // We drive the queue with source PTS, not a real clock, so disable
        // libplacebo's drift compensation (matches vo_gpu_next).
        qparams.drift_compensation = 0;
#endif

        // Seek/reset can leave us slightly before the first queued frame (the
        // demuxer PTS does not start at 0). pl_queue requires a monotonic PTS,
        // so clamp to the first available frame instead of dropping it.
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
            goto done;
        case PL_QUEUE_EOF:
            abort(); // we never signal EOF
        case PL_QUEUE_MORE:
            // Expected near the start and end of a file, so keep it quiet.
            MP_DBG(p, "Render queue underrun.\n");
            break;
        case PL_QUEUE_OK:
            break;
        }
    } else {
        // Redraw without a current frame: reuse the last PTS. This never
        // advances the queue, which is what we want for OSD-only redraws.
        struct pl_queue_params qparams = *pl_queue_params(.pts = p->last_pts);
        pl_queue_update(p->queue, &mix, &qparams);
    }

    // Apply the source crop/rotation to every mix frame. Must happen before
    // pl_render_image_mix(): libplacebo derives the crop from the reference
    // plane's texture when none is set, which is unset for hwdec frames until
    // ->acquire() runs. vo_gpu_next does the same.
    for (int i = 0; i < mix.num_frames; i++) {
        struct pl_frame *image = (struct pl_frame *)mix.frames[i];
        struct mp_image *mpi = image->user_data;
        apply_crop(image, p->current_src, mpi->params.w, mpi->params.h);
    }

    // Signatures must be unique per queue entry and stable across draws. Use
    // the frame ID (not the mp_image pointer, which allocators recycle) mixed
    // with the OSD version so that an OSD-only redraw invalidates the cache.
    // The array is clamped rather than asserted: with a custom frame mixer the
    // radius could grow beyond this bound, and overflowing the stack is not an
    // acceptable failure mode.
    uint64_t signatures[32];
    // Overlays are rebuilt every draw, so bump the OSD generation to force
    // libplacebo to recompute signatures and re-composite the OSD.
    p->osd_sync++;
    int num_sig = MPMIN(mix.num_frames, MP_ARRAY_SIZE(signatures));
    for (int i = 0; i < num_sig; i++) {
        struct pl_frame *image = (struct pl_frame *)mix.frames[i];
        struct mp_image *mpi = image->user_data;
        struct frame_priv *fp = mpi->priv;
        fp->osd_sync = p->osd_sync;
        signatures[i] = fp->signature ^ (fp->osd_sync << 48);
    }
    mix.signatures = signatures;
    mix.num_frames = num_sig;

    // The first frame in the mix provides the colorspace for the OSD. It is
    // NULL when there is no video, in which case overlays render on black.
    struct mp_image *ref = mix.num_frames > 0 ? mix.frames[0]->user_data : NULL;
    update_overlays(p, p->osd_res, 0, PL_OVERLAY_COORDS_DST_FRAME,
                    &p->osd_state, &target, ref);

    struct pl_render_params params = p->pars->params;

    struct pl_color_adjustment color_adj;
    struct mp_csp_params cparams = MP_CSP_PARAMS_DEFAULTS;
    mp_csp_equalizer_state_get(p->video_eq, &cparams);
    color_adj.brightness = cparams.brightness;
    color_adj.contrast   = cparams.contrast;
    color_adj.hue        = cparams.hue;
    color_adj.saturation = cparams.saturation;
    color_adj.gamma      = cparams.gamma;
    params.color_adjustment = &color_adj;

    params.preserve_mixing_cache = p->opts->inter_preserve && !(frame && frame->still);
    if (frame && frame->still)
        params.frame_mixer = NULL;
    // Cache a frame only when it will actually be reused (repeated/redrawn),
    // matching vo_gpu_next: will_redraw || still.
    bool cache_frame = frame && (will_redraw || frame->still);
    params.skip_caching_single_frame = !cache_frame;

    if (!pl_render_image_mix(p->renderer, &mix, &target, &params))
        MP_ERR(p, "Rendering failed.\n");

done:
    ;
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
    // A screenshot may be taken between frames; bump the OSD version so the
    // renderer treats it as a distinct frame.
    p->osd_sync++;
    update_overlays(p, osd, 0, PL_OVERLAY_COORDS_DST_FRAME, &p->osd_state, &target,
                    frame->current);

    // Use the same render options as playback.
    m_config_cache_update(p->gl_opts_cache);
    m_config_cache_update(p->opts_cache);
    struct pl_render_params params = p->pars->params;
    params.skip_caching_single_frame = true;
    params.frame_mixer = NULL;
    if (!pl_render_image(p->renderer, &source, &target, &params)) {
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
    // Parameters are propagated through the frames themselves (map_frame()
    // reads mpi->params), so there is nothing to store here beyond refreshing
    // the options that affect rendering.
    m_config_cache_update(p->opts_cache);
}

void pl_video_resize(struct pl_video *p, const struct mp_rect *src,
                     const struct mp_rect *dst, const struct mp_osd_res *osd)
{
    if (src)
        p->current_src = *src;
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
    // Defer the actual flush to the next push: the PTS counter must be reset
    // at the same time the queue is, so that the clamp logic can recover the
    // first frame after a seek.
    p->want_reset = true;
}

