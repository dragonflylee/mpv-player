/* Copyright (C) 2018 the mpv developers
 *
 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 * ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 * OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */

#ifndef MPV_CLIENT_API_RENDER_GL_H_
#define MPV_CLIENT_API_RENDER_GL_H_

#include <psp2/gxm.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * For initializing the mpv deko3d state via MPV_RENDER_PARAM_DEKO3D_INIT_PARAMS.
 */
typedef struct mpv_gxm_init_params {
    /**
     * The gxm context and shader patcher that will be used in subsequent operation.
     */
    SceGxmContext *context;
    SceGxmShaderPatcher *shader_patcher;
    /**
     * Fragment Uniform buffer index.
     * sceGxmSetFragmentUniformBuffer(...) will use this index to set the fragment uniform buffer.
     * If you set `-Dvitashark=disabled` when building libmpv, buffer_index will be fixed to 0
     * Please make sure it does not conflict with other parts of your application
     */
    int buffer_index;
    /**
     * multisample mode of mpv shader.
     * This should set to the same value as the one used in other parts of your application
     */
    SceGxmMultisampleMode msaa;
} mpv_gxm_init_params;

/**
 * For MPV_RENDER_PARAM_GXM_FBO.
 */
typedef struct mpv_gxm_fbo {
    /**
     * Set the framebuffer to render to.
     * if any object is NULL, the default framebuffer will be used.
     * When using the default framebuffer, make sure mpv_render_context_render(...) runs between sceGxmBeginScene and sceGxmEndScene
     * When using a custom framebuffer, make sure mpv_render_context_render(...) runs outside sceGxmBeginScene and sceGxmEndScene
     */
    SceGxmRenderTarget *render_target;
    SceGxmColorSurface *color_surface;
    SceGxmDepthStencilSurface *depth_stencil_surface;
    /**
     * Valid dimensions. This must refer to the size of the framebuffer. This
     * must always be set.
     */
    int w, h;
    /**
     * Underlying texture internal format. This must always be set.
     */
    SceGxmTextureFormat format;
} mpv_gxm_fbo;

#ifdef __cplusplus
}
#endif // __cplusplus

#endif