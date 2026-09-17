#pragma once

#include <libplacebo/common.h>
#include <libplacebo/gpu.h>
#include <libplacebo/renderer.h>
#include <libplacebo/shaders/lut.h>
#include <libplacebo/utils/upload.h>

#include "options/m_option.h"
#include "video/img_format.h"

// Shared between vo_gpu_next and the 'gpu-next' libmpv render backend.
extern const struct m_sub_options gl_next_conf;

struct user_lut {
    char *opt;
    char *path;
    int type;
    struct pl_custom_lut *lut;
};

struct gl_next_opts {
    bool delayed_peak;
    int sub_hdr_peak;
    int image_subs_hdr_peak;
    int border_background;
    float background_blur_radius;
    float corner_rounding;
    bool inter_preserve;
    struct user_lut lut;
    struct user_lut image_lut;
    struct user_lut target_lut;
    int target_hint;
    int target_hint_mode;
    bool target_hint_strict;
    char **raw_opts;
};

// Translate an mpv image format into a libplacebo plane description. Returns
// the number of planes, or 0 if the format is unsupported. With `use_uint`,
// PL_FMT_UINT is used instead of PL_FMT_UNORM for the component type.
int plane_data_from_imgfmt(struct pl_plane_data out_data[4],
                           struct pl_bit_encoding *out_bits,
                           enum mp_imgfmt imgfmt, bool use_uint);

// Translate mpv's rotated/flipped display rect into libplacebo's unrotated
// `pl_frame.crop` convention.
void apply_crop(struct pl_frame *frame, struct mp_rect crop,
                   int width, int height);
