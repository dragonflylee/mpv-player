#include <switch.h>

#include <string.h>

#include "misc/language.h"
#include "mpv_talloc.h"
#include "osdep/strnlen.h"

char **mp_get_user_langs(void)
{
    // Read the system language from Horizon OS; env vars are not available.
    // LanguageCode is a u64 holding a NUL-terminated tag like "en-US"
    // (not the SetLanguage enum).
    // https://switchbrew.org/wiki/Settings_services#LanguageCode
    u64 languageCode = 0;
    if (R_FAILED(setGetSystemLanguage(&languageCode)))
        return NULL;

    // Bound the length in case it is not NUL-terminated within the u64.
    const char *lang = (const char *)&languageCode;
    size_t len = strnlen(lang, sizeof(languageCode));
    if (!len)
        return NULL;

    char **ret = talloc_array_ptrtype(NULL, ret, 2);
    ret[0] = talloc_strndup(ret, lang, len);
    ret[1] = NULL;

    return ret;
}
