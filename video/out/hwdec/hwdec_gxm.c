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
 * License along with mpv.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <psp2/gxm.h>
#include <libavutil/macros.h>
#include "video/out/gpu/hwdec.h"

struct priv_owner {
    size_t gc_count;
    size_t gc_index;
    struct mp_image **gc_list;
};

struct gxm_priv {
};

static int init(struct ra_hwdec *hw) {
    struct priv_owner *priv = hw->priv;

    // TODO: gc_count should be >= `display buffer count` of user's application
    priv->gc_count = 3;
    priv->gc_index = 0;
    priv->gc_list = talloc_zero_array(NULL, struct mp_image*, priv->gc_count);

    return 0;
}

static void uninit(struct ra_hwdec *hw) {
    struct priv_owner *priv = hw->priv;

    for (int i = 0; i < priv->gc_count; i++) {
        if (priv->gc_list[i]) {
            mp_image_unrefp(&priv->gc_list[i]);
        }
    }

    talloc_free(priv->gc_list);
}

static int mapper_init(struct ra_hwdec_mapper *mapper) {
    if (!mapper->ra->glsl_gxm) {
        MP_ERR(mapper, "Only GXM is not supported\n");
        return -1;
    }
    mapper->dst_params = mapper->src_params;

    struct ra_tex *tex = NULL;
    struct ra_imgfmt_desc desc;
    if (!ra_get_imgfmt_desc(mapper->ra, mapper->dst_params.imgfmt, &desc)) {
        MP_ERR(mapper, "Unsupported format: %s\n", mp_imgfmt_to_name(mapper->dst_params.imgfmt));
        return -1;
    }

    desc.num_planes = 1;

    tex = mapper->tex[0] = talloc_zero(mapper, struct ra_tex);
    if (!tex)
        return -1;

    tex->params = (struct ra_tex_params) {
        .dimensions = 2,
        .w          = FFMAX(FFALIGN(mapper->dst_params.w, 16), 64),
        .h          = FFMAX(FFALIGN(mapper->dst_params.h, 16), 64),
        .d          = 1,
        .format     = desc.planes[0],
        .render_src = true,
        .src_linear = true,
    };

    return 0;
}

static void mapper_uninit(struct ra_hwdec_mapper *mapper) {
    (void)mapper;
}

struct ra_tex_gxm {
    SceGxmTexture gxm_tex;
    uint8_t *tex_data;
    SceUID data_UID;
    SceGxmTextureFormat format;
    int bpp;
    int stride;
};

static int mapper_map(struct ra_hwdec_mapper *mapper) {
    struct priv_owner *priv = mapper->owner->priv;
    uint8_t* tex_data = mapper->src->bufs[0]->data;
    size_t tex_size = mapper->src->bufs[0]->size;

    struct ra_tex *tex = mapper->tex[0];
    struct ra_tex_gxm *tex_gxm = tex->priv = talloc_zero(mapper->tex[0], struct ra_tex_gxm);
    tex_gxm->tex_data = tex_data;

    sceGxmMapMemory(tex_data, tex_size, SCE_GXM_MEMORY_ATTRIB_READ);
    sceGxmTextureInitLinear(&tex_gxm->gxm_tex, tex_data, SCE_GXM_TEXTURE_FORMAT_YVU420P2_CSC0, tex->params.w, tex->params.h, 0);
    sceGxmTextureSetMinFilter(&tex_gxm->gxm_tex, SCE_GXM_TEXTURE_FILTER_LINEAR);
    sceGxmTextureSetMagFilter(&tex_gxm->gxm_tex, SCE_GXM_TEXTURE_FILTER_LINEAR);

    // garbage collect
    struct mp_image *new_img = mp_image_new_ref(mapper->src);
    struct mp_image *old_img = priv->gc_list[priv->gc_index];
    if (old_img) {
        tex_data = old_img->bufs[0]->data;
        sceGxmUnmapMemory(tex_data);
        mp_image_unrefp(&old_img);
    }
    priv->gc_list[priv->gc_index] = new_img;
    priv->gc_index = (priv->gc_index + 1) % priv->gc_count;

    return 0;
}

static void mapper_unmap(struct ra_hwdec_mapper *mapper) {
    (void)mapper;
}

const struct ra_hwdec_driver ra_hwdec_gxm = {
        .name = "gxm",
        .priv_size = sizeof(struct priv_owner),
        .imgfmts = {IMGFMT_VITA_NV12, 0},
        .init = init,
        .uninit = uninit,
        .mapper = &(const struct ra_hwdec_mapper_driver){
                .priv_size = sizeof(struct gxm_priv),
                .init = mapper_init,
                .uninit = mapper_uninit,
                .map = mapper_map,
                .unmap = mapper_unmap,
        },
};