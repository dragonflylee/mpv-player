#include <psp2/apputil.h>
#include <psp2/system_param.h>

#include "misc/language.h"
#include "mpv_talloc.h"

static void psv_apputil_init(void)
{
    static bool inited;
    if (inited)
        return;
    inited = true;

    SceAppUtilInitParam initParam = {0};
    SceAppUtilBootParam bootParam = {0};
    sceAppUtilInit(&initParam, &bootParam);
}

char **mp_get_user_langs(void)
{
    psv_apputil_init();

    int lang = SCE_SYSTEM_PARAM_LANG_ENGLISH_US;
    if (sceAppUtilSystemParamGetInt(SCE_SYSTEM_PARAM_ID_LANG, &lang) < 0)
        return NULL;

    const char *tag;
    switch (lang) {
    case SCE_SYSTEM_PARAM_LANG_JAPANESE:       tag = "ja";    break;
    case SCE_SYSTEM_PARAM_LANG_ENGLISH_US:     tag = "en-US"; break;
    case SCE_SYSTEM_PARAM_LANG_FRENCH:         tag = "fr";    break;
    case SCE_SYSTEM_PARAM_LANG_SPANISH:        tag = "es";    break;
    case SCE_SYSTEM_PARAM_LANG_GERMAN:         tag = "de";    break;
    case SCE_SYSTEM_PARAM_LANG_ITALIAN:        tag = "it";    break;
    case SCE_SYSTEM_PARAM_LANG_DUTCH:          tag = "nl";    break;
    case SCE_SYSTEM_PARAM_LANG_PORTUGUESE_PT:  tag = "pt-PT"; break;
    case SCE_SYSTEM_PARAM_LANG_RUSSIAN:        tag = "ru";    break;
    case SCE_SYSTEM_PARAM_LANG_KOREAN:         tag = "ko";    break;
    case SCE_SYSTEM_PARAM_LANG_CHINESE_T:      tag = "zh-Hant"; break;
    case SCE_SYSTEM_PARAM_LANG_CHINESE_S:      tag = "zh-Hans"; break;
    case SCE_SYSTEM_PARAM_LANG_FINNISH:        tag = "fi";    break;
    case SCE_SYSTEM_PARAM_LANG_SWEDISH:        tag = "sv";    break;
    case SCE_SYSTEM_PARAM_LANG_DANISH:         tag = "da";    break;
    case SCE_SYSTEM_PARAM_LANG_NORWEGIAN:      tag = "no";    break;
    case SCE_SYSTEM_PARAM_LANG_POLISH:         tag = "pl";    break;
    case SCE_SYSTEM_PARAM_LANG_PORTUGUESE_BR:  tag = "pt-BR"; break;
    case SCE_SYSTEM_PARAM_LANG_ENGLISH_GB:     tag = "en-GB"; break;
    case SCE_SYSTEM_PARAM_LANG_TURKISH:        tag = "tr";    break;
    default:                                   tag = "en-US"; break;
    }

    char **ret = talloc_array_ptrtype(NULL, ret, 2);
    ret[0] = talloc_strdup(ret, tag);
    ret[1] = NULL;

    return ret;
}
