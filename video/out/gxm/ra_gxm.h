#pragma once

#include <stdbool.h>
#include <psp2/gxm.h>

#include "video/out/gpu/ra.h"
#include "video/out/gpu/spirv.h"

#define MAX_VERTEX_POINTS 48

typedef struct GxmShaderProgram {
    SceGxmShaderPatcherId vert_id;
    SceGxmShaderPatcherId frag_id;

    SceGxmVertexProgram *vert;
    SceGxmFragmentProgram *frag;

    SceGxmProgram *vert_gxp;
    SceGxmProgram *frag_gxp;
} GxmShaderProgram;

struct mp_gxm_clear_vertex {
    float x, y;
};

struct ra_gxm {
    SceGxmContext *context;
    SceGxmShaderPatcher *shader_patcher;

    // clear shader
    GxmShaderProgram clearProg;
    const SceGxmProgramParameter *clearParam;
    SceUID clearVerticesUid;
    struct mp_gxm_clear_vertex *clearVertices;

    // shared linear indices
    SceUID linearIndicesUid;
    unsigned short *linearIndices;

    // uniform buffer index
    int buffer_index;

    // Application multisample mode
    SceGxmMultisampleMode msaa;
};

struct ra_tex_gxm {
    SceGxmTexture gxm_tex;
    uint8_t *tex_data;
    SceUID data_UID;
    SceGxmTextureFormat format;
    int bpp;
    int stride;
};

struct gxm_format {
    const char *name;
    int components;
    int bytes;
    int bits[4];
    SceGxmTextureFormat format;
    enum ra_ctype ctype;
    bool renderable, linear_filter, storable, ordered;
};

// Create an RA instance.
struct ra *ra_gxm_create(struct mp_log *log, SceGxmContext *context, SceGxmShaderPatcher *shader_patcher,
                         int buffer_index, SceGxmMultisampleMode msaa);

// True if the RA instance was created with ra_gxm_create()
bool ra_is_gxm(struct ra *ra);
