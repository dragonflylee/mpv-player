// Taken from: https://github.com/fish47/mpv-vita/blob/vita/audio/out/ao_vita.c

#include "internal.h"
#include "audio/format.h"
#include "common/msg.h"
#include "osdep/timer.h"
#include "osdep/threads.h"

#include <psp2/audioout.h>
#include <malloc.h>
#include <errno.h>

#define FRAME_COUNT             2
#define FRAME_ALIGN             64

#define BUFFER_SAMPLE_COUNT     1024

// How long uninit() waits for the play thread before closing the port, which
// makes a blocking sceAudioOutOutput() return immediately.
#define THREAD_JOIN_TIMEOUT_US  (500 * 1000LL)

#define FLAG_PAUSED             (1)
#define FLAG_TERMINATED         (1 << 1)

#define INIT_FIELD_BASE         (1)
#define INIT_FIELD_THREAD       (1 << 2)

struct priv {
    pthread_mutex_t lock;
    pthread_cond_t wakeup;
    pthread_t play_thread;
    int audio_ctx;         // port id, -1 when not open
    void *frame_base;
    int init_fields;
    int run_flags;
    bool thread_exited;
};

static bool audio_init(struct ao *ao, int *ctx, int samples, int freq, int channels) {
    SceAudioOutPortType type = (freq < 48000)
                               ? SCE_AUDIO_OUT_PORT_TYPE_BGM
                               : SCE_AUDIO_OUT_PORT_TYPE_MAIN;
    SceAudioOutMode mode = (channels == 1)
                           ? SCE_AUDIO_OUT_MODE_MONO
                           : SCE_AUDIO_OUT_MODE_STEREO;

    int port = sceAudioOutOpenPort(type, samples, freq, mode);
    if (port < 0) {
        MP_ERR(ao, "could not open audio port: 0x%08x\n", port);
        return false;
    }

    int vols[2] = {SCE_AUDIO_OUT_MAX_VOL, SCE_AUDIO_OUT_MAX_VOL};
    SceAudioOutChannelFlag flags = SCE_AUDIO_VOLUME_FLAG_L_CH | SCE_AUDIO_VOLUME_FLAG_R_CH;
    sceAudioOutSetVolume(port, flags, vols);

    *ctx = port;
    return true;
}

static void audio_uninit(int *ctx) {
    if (*ctx >= 0) {
        sceAudioOutReleasePort(*ctx);
        *ctx = -1;
    }
}

static const int supported_samplerates[] = {
        8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000
};

static int choose_samplerate(int samplerate)
{
    for (int i = 0; i < MP_ARRAY_SIZE(supported_samplerates); ++i) {
        int sr = supported_samplerates[i];
        if (sr >= samplerate)
            return sr;
    }

    int max_idx = MP_ARRAY_SIZE(supported_samplerates) - 1;
    return supported_samplerates[max_idx];
}

static void set_thread_state(struct ao *ao, int flag_bit, bool set)
{
    struct priv *priv = ao->priv;
    pthread_mutex_lock(&priv->lock);
    int *flags = &priv->run_flags;
    *flags = set ? (*flags | flag_bit) : (*flags & ~flag_bit);
    pthread_cond_broadcast(&priv->wakeup);
    pthread_mutex_unlock(&priv->lock);
}

static size_t calculate_frame_size(struct ao *ao)
{
    // s16 is 2 bytes per sample
    size_t result = BUFFER_SAMPLE_COUNT;
    result *= 2;
    result *= ao->channels.num;
    return result;
}

static void *thread_run(void *arg)
{
    struct ao *ao = arg;
    struct priv *priv = ao->priv;

    mpthread_set_name("vita-audio");

    void *frame = priv->frame_base;
    void *other = (uint8_t *)priv->frame_base + calculate_frame_size(ao);
    bool queued = false;

    pthread_mutex_lock(&priv->lock);
    while (!(priv->run_flags & FLAG_TERMINATED)) {
        if (priv->run_flags & FLAG_PAUSED) {
            // Drop the queued buffer so resuming starts with silence.
            queued = false;
            pthread_cond_wait(&priv->wakeup, &priv->lock);
            continue;
        }

        pthread_mutex_unlock(&priv->lock);

        // Poll the port so FLAG_TERMINATED stays observable between driver
        // calls; a closed port ends the wait too.
        while (queued) {
            pthread_mutex_lock(&priv->lock);
            bool terminate = priv->run_flags & FLAG_TERMINATED;
            pthread_mutex_unlock(&priv->lock);
            if (terminate || sceAudioOutGetRestSample(priv->audio_ctx) <= 0)
                break;
            mp_sleep_us(2000);
        }

        // The port holds one buffer at a time, so a buffer written now is
        // heard after its own length.
        int64_t out_time = mp_time_us() + 1000000LL *
                           BUFFER_SAMPLE_COUNT / ao->samplerate;
        ao_read_data(ao, &frame, BUFFER_SAMPLE_COUNT, out_time);
        MPSWAP(void *, frame, other);

        // Blocks for at most one buffer period; uninit() may close the port
        // meanwhile, making this fail right away.
        queued = sceAudioOutOutput(priv->audio_ctx, frame) >= 0;

        pthread_mutex_lock(&priv->lock);
    }
    priv->thread_exited = true;
    pthread_cond_broadcast(&priv->wakeup);
    pthread_mutex_unlock(&priv->lock);

    return NULL;
}

static bool wait_thread_exit(struct ao *ao, int64_t timeout_us)
{
    struct priv *priv = ao->priv;
    int64_t deadline = mp_time_us() + timeout_us;

    pthread_mutex_lock(&priv->lock);
    while (!priv->thread_exited) {
        int64_t left = deadline - mp_time_us();
        if (left <= 0)
            break;
        struct timespec ts = mp_time_us_to_timespec(mp_time_us() + left);
        if (pthread_cond_timedwait(&priv->wakeup, &priv->lock, &ts) == ETIMEDOUT)
            break;
    }
    bool done = priv->thread_exited;
    pthread_mutex_unlock(&priv->lock);
    return done;
}

static void uninit(struct ao *ao)
{
    struct priv *priv = ao->priv;

    if (priv->init_fields & INIT_FIELD_THREAD) {
        set_thread_state(ao, FLAG_TERMINATED, true);

        // Closing the port unblocks a thread stuck in sceAudioOutOutput().
        if (!wait_thread_exit(ao, THREAD_JOIN_TIMEOUT_US)) {
            audio_uninit(&priv->audio_ctx);
            wait_thread_exit(ao, THREAD_JOIN_TIMEOUT_US);
        }

        pthread_join(priv->play_thread, NULL);
        priv->init_fields &= ~INIT_FIELD_THREAD;
    }

    if (priv->init_fields & INIT_FIELD_BASE) {
        pthread_mutex_destroy(&priv->lock);
        pthread_cond_destroy(&priv->wakeup);
        priv->init_fields &= ~INIT_FIELD_BASE;
    }

    audio_uninit(&priv->audio_ctx);

    if (priv->frame_base)
        free(priv->frame_base);

    memset(priv, 0, sizeof(struct priv));
    priv->audio_ctx = -1;
}

static int init(struct ao *ao)
{
    struct priv *priv = ao->priv;

    // conform to hardware/API restrictions
    ao->format = AF_FORMAT_S16;
    ao->samplerate = choose_samplerate(ao->samplerate);
    if (ao->channels.num > 2)
        mp_chmap_from_channels(&ao->channels, 2);

    memset(priv, 0, sizeof(struct priv));
    priv->audio_ctx = -1;

    if (pthread_mutex_init(&priv->lock, NULL)) {
        MP_ERR(ao, "could not initialize mutex\n");
        goto error;
    }
    if (pthread_cond_init(&priv->wakeup, NULL)) {
        MP_ERR(ao, "could not initialize condition variable\n");
        pthread_mutex_destroy(&priv->lock);
        goto error;
    }
    priv->init_fields |= INIT_FIELD_BASE;
    set_thread_state(ao, FLAG_PAUSED, true);

    if (!audio_init(ao, &priv->audio_ctx, BUFFER_SAMPLE_COUNT, ao->samplerate,
                    ao->channels.num))
        goto error;

    // The thread touches these buffers, so allocate before it starts.
    size_t frame_size = calculate_frame_size(ao);
    priv->frame_base = memalign(FRAME_ALIGN, frame_size * FRAME_COUNT);
    if (!priv->frame_base)
        goto error;
    memset(priv->frame_base, 0, frame_size * FRAME_COUNT);

    if (pthread_create(&priv->play_thread, NULL, thread_run, ao)) {
        MP_ERR(ao, "could not create play thread\n");
        goto error;
    }
    priv->init_fields |= INIT_FIELD_THREAD;

    // ensure buffer capacity to avoid underrun on audio thread
    ao->device_buffer = BUFFER_SAMPLE_COUNT * 2;

    return 1;

error:
    uninit(ao);
    return -1;
}

static void reset(struct ao *ao)
{
    set_thread_state(ao, FLAG_PAUSED, true);
}

static void start(struct ao *ao)
{
    set_thread_state(ao, FLAG_PAUSED, false);
}

const struct ao_driver audio_out_vita = {
        .description = "Vita Audio",
        .name = "vita",
        .init = init,
        .uninit = uninit,
        .reset = reset,
        .start = start,
        .priv_size = sizeof(struct priv),
};
