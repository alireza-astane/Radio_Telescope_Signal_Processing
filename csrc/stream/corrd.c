/*
 * corrd: real-time FX correlator daemon.
 *
 *   receivers --TCP--> [rx thread per antenna] --> per-antenna ring buffers
 *                                                      |
 *              [correlator thread] waits until frame `seq` is present for
 *              every antenna and unpacks it (int8 -> float) into the current
 *              integration; every --integrate frames it runs the F-step (FFT,
 *              OpenMP over antennas) and one AVX2 X-step over the whole
 *              integration, and writes a visibility dump.
 *
 * A receiver blocks (TCP back-pressure) when it is --ring frames ahead of the
 * correlator, so nothing is dropped silently; frames whose sequence numbers
 * skip are counted as lost.  Prints throughput, per-stage time and the
 * real-time factor relative to --nominal-rate.
 *
 * Replaces legacy/client4.c and legacy/client5.c, whose receive threads wrote
 * into shared buffers without holding the lock and whose FFT plans were
 * recreated for every block.
 */
#define _GNU_SOURCE
#include "../fft.h"
#include "../fxcorr.h"
#include "frame.h"

#include <errno.h>
#include <getopt.h>
#include <math.h>
#include <netinet/in.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

typedef struct {
    int n_ant, nchan, frame, ring, integrate;
    const char *out_path;
    const char *kernel;
    double nominal_msps;
} config;

typedef struct {
    int8_t *data;     /* [ring][2 * frame] */
    int64_t *seq_in;  /* sequence number held by each slot, -1 = empty */
    int connected;
    int done;         /* connection closed */
    uint64_t lost;    /* sequence gaps */
} antenna_ring;

static config cfg;
static antenna_ring *rings;
static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t data_ready = PTHREAD_COND_INITIALIZER;
static pthread_cond_t space_free = PTHREAD_COND_INITIALIZER;
static int64_t next_seq = 0; /* next frame the correlator will consume */

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static int recv_all(int fd, void *buf, size_t len)
{
    char *p = buf;
    while (len > 0) {
        ssize_t n = recv(fd, p, len, MSG_WAITALL);
        if (n == 0)
            return 0;
        if (n < 0) {
            if (errno == EINTR)
                continue;
            return -1;
        }
        p += n;
        len -= (size_t)n;
    }
    return 1;
}

static void *rx_thread(void *arg)
{
    int fd = (int)(intptr_t)arg;
    size_t payload = 2 * (size_t)cfg.frame;
    int8_t *tmp = malloc(payload);
    rt_frame_header h;
    int ant = -1;
    int64_t expect = 0;

    while (recv_all(fd, &h, sizeof(h)) == 1) {
        if (h.magic != RT_FRAME_MAGIC || h.n_samples != (uint32_t)cfg.frame || h.antenna >= cfg.n_ant) {
            fprintf(stderr, "corrd: bad frame header (magic %08x, n_samples %u, antenna %u)\n", h.magic,
                    h.n_samples, h.antenna);
            break;
        }
        if (recv_all(fd, tmp, payload) != 1)
            break;
        if (ant < 0) {
            ant = h.antenna;
            pthread_mutex_lock(&lock);
            rings[ant].connected = 1;
            pthread_mutex_unlock(&lock);
        }
        int64_t seq = (int64_t)h.seq;
        pthread_mutex_lock(&lock);
        if (seq > expect)
            rings[ant].lost += (uint64_t)(seq - expect);
        expect = seq + 1;
        while (seq >= next_seq + cfg.ring) /* back-pressure: wait for a free slot */
            pthread_cond_wait(&space_free, &lock);
        if (seq >= next_seq) {
            int slot = (int)(seq % cfg.ring);
            memcpy(rings[ant].data + (size_t)slot * payload, tmp, payload);
            rings[ant].seq_in[slot] = seq;
            pthread_cond_broadcast(&data_ready);
        }
        pthread_mutex_unlock(&lock);
    }

    pthread_mutex_lock(&lock);
    if (ant >= 0)
        rings[ant].done = 1;
    pthread_cond_broadcast(&data_ready);
    pthread_mutex_unlock(&lock);
    free(tmp);
    close(fd);
    return NULL;
}

/* Wait until frame `seq` is present for every antenna; 0 if the stream ended. */
static int wait_frame(int64_t seq)
{
    int slot = (int)(seq % cfg.ring);
    pthread_mutex_lock(&lock);
    for (;;) {
        int ready = 1, ended = 0;
        for (int a = 0; a < cfg.n_ant; a++) {
            if (rings[a].seq_in[slot] != seq) {
                ready = 0;
                if (rings[a].done)
                    ended = 1;
            }
        }
        if (ready || ended) {
            pthread_mutex_unlock(&lock);
            return ready;
        }
        pthread_cond_wait(&data_ready, &lock);
    }
}

static void release_frame(int64_t seq)
{
    pthread_mutex_lock(&lock);
    next_seq = seq + 1;
    pthread_cond_broadcast(&space_free);
    pthread_mutex_unlock(&lock);
}

static void usage(const char *prog)
{
    fprintf(stderr,
            "usage: %s --n-ant N [--port P] [--nchan C] [--frame SAMPLES] [--integrate FRAMES]\n"
            "          [--ring FRAMES] [--threads T (default: half the CPUs)] [--kernel scalar|avx2|avx2_omp]\n"
            "          [--nominal-rate MSPS] [--out FILE]\n",
            prog);
}

int main(int argc, char **argv)
{
    cfg = (config){.n_ant = 0, .nchan = 64, .frame = 4096, .ring = 64, .integrate = 16,
                   .out_path = NULL, .kernel = "avx2_omp", .nominal_msps = 2.4};
    int port = 5000, threads = 0;
    static struct option opts[] = {
        {"n-ant", required_argument, 0, 'n'},     {"port", required_argument, 0, 'p'},
        {"nchan", required_argument, 0, 'c'},     {"frame", required_argument, 0, 'f'},
        {"integrate", required_argument, 0, 'k'}, {"ring", required_argument, 0, 'r'},
        {"threads", required_argument, 0, 't'},   {"kernel", required_argument, 0, 'K'},
        {"nominal-rate", required_argument, 0, 'R'}, {"out", required_argument, 0, 'o'},
        {0, 0, 0, 0}};
    int o;
    while ((o = getopt_long(argc, argv, "", opts, NULL)) != -1) {
        switch (o) {
        case 'n': cfg.n_ant = atoi(optarg); break;
        case 'p': port = atoi(optarg); break;
        case 'c': cfg.nchan = atoi(optarg); break;
        case 'f': cfg.frame = atoi(optarg); break;
        case 'k': cfg.integrate = atoi(optarg); break;
        case 'r': cfg.ring = atoi(optarg); break;
        case 't': threads = atoi(optarg); break;
        case 'K': cfg.kernel = optarg; break;
        case 'R': cfg.nominal_msps = atof(optarg); break;
        case 'o': cfg.out_path = optarg; break;
        default: usage(argv[0]); return 2;
        }
    }
    if (cfg.n_ant < 2 || cfg.frame % cfg.nchan != 0 || cfg.ring < 2 || cfg.integrate < 1) {
        usage(argv[0]);
        return 2;
    }
    void (*xstep)(const float *, int, int, int, float *) = fx_xstep_avx2_omp;
    if (strcmp(cfg.kernel, "scalar") == 0)
        xstep = fx_xstep_scalar;
    else if (strcmp(cfg.kernel, "avx2") == 0)
        xstep = fx_xstep_avx2;
    if (xstep != fx_xstep_scalar && !fx_cpu_has_avx2()) {
        fprintf(stderr, "corrd: CPU lacks AVX2/FMA, falling back to scalar kernel\n");
        xstep = fx_xstep_scalar;
    }
    /* Default: half the hardware threads for compute, leaving the rest to the
     * per-receiver socket threads (oversubscribing stalls the I/O and builds lag). */
    fx_set_threads(threads > 0 ? threads : (fx_max_threads() + 1) / 2);

    fft_plan plan;
    if (fft_plan_init(&plan, cfg.nchan) != 0) {
        fprintf(stderr, "corrd: --nchan must be a power of two\n");
        return 2;
    }

    const int n_bl = fx_n_baselines(cfg.n_ant);
    const int n_spec = cfg.frame / cfg.nchan;
    const size_t payload = 2 * (size_t)cfg.frame;
    rings = calloc(cfg.n_ant, sizeof(antenna_ring));
    for (int a = 0; a < cfg.n_ant; a++) {
        rings[a].data = malloc((size_t)cfg.ring * payload);
        rings[a].seq_in = malloc(sizeof(int64_t) * cfg.ring);
        for (int s = 0; s < cfg.ring; s++)
            rings[a].seq_in[s] = -1;
    }
    /* One integration = `integrate` frames.  Their spectra are laid out per antenna
     * back to back ([ant][integrate * n_spec][nchan]), so a single X-step call over
     * all of them yields the time-averaged visibilities directly. */
    const size_t ant_span = (size_t)cfg.integrate * cfg.frame; /* complex samples per antenna */
    const size_t spec_bytes = sizeof(float) * 2 * (size_t)cfg.n_ant * ant_span;
    float *spectra = aligned_alloc(64, (spec_bytes + 63) / 64 * 64);
    float *vis = malloc(sizeof(float) * 2 * (size_t)n_bl * cfg.nchan);

    FILE *out = NULL;
    if (cfg.out_path && !(out = fopen(cfg.out_path, "wb"))) {
        perror(cfg.out_path);
        return 1;
    }

    /* Listen and accept one connection per antenna. */
    int lfd = socket(AF_INET, SOCK_STREAM, 0);
    int one = 1;
    setsockopt(lfd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    struct sockaddr_in addr = {.sin_family = AF_INET, .sin_port = htons(port), .sin_addr.s_addr = INADDR_ANY};
    if (bind(lfd, (struct sockaddr *)&addr, sizeof(addr)) < 0 || listen(lfd, cfg.n_ant) < 0) {
        perror("corrd: bind/listen");
        return 1;
    }
    fprintf(stderr, "corrd: listening on port %d for %d antennas (%d baselines, %d channels, kernel %s, %d threads)\n",
            port, cfg.n_ant, n_bl, cfg.nchan, cfg.kernel, fx_max_threads());

    pthread_t *th = calloc(cfg.n_ant, sizeof(pthread_t));
    for (int a = 0; a < cfg.n_ant; a++) {
        int fd = accept(lfd, NULL, NULL);
        if (fd < 0) {
            perror("corrd: accept");
            return 1;
        }
        pthread_create(&th[a], NULL, rx_thread, (void *)(intptr_t)fd);
    }
    close(lfd);

    const float scale = 1.0f / sqrtf((float)cfg.nchan); /* unitary FFT, matches numpy norm="ortho" */
    const int n_spec_batch = cfg.integrate * n_spec;
    double t_start = 0, t_f = 0, t_x = 0;
    int64_t frames = 0, dumps = 0, batch_first = 0, discarded = 0;
    int in_batch = 0;

    for (int64_t seq = 0;; seq++) {
        if (!wait_frame(seq)) {
            discarded = in_batch; /* incomplete final integration */
            break;
        }
        if (seq == 0)
            t_start = now_s();

        /* Unpack int8 -> float straight into this frame's place in the batch,
         * then hand the ring slot back to the receivers. */
        double t0 = now_s();
        int slot = (int)(seq % cfg.ring);
        for (int a = 0; a < cfg.n_ant; a++) {
            const int8_t *src = rings[a].data + (size_t)slot * payload;
            float *dst = spectra + 2 * ((size_t)a * ant_span + (size_t)in_batch * cfg.frame);
            for (int i = 0; i < 2 * cfg.frame; i++)
                dst[i] = (float)src[i] * scale;
        }
        release_frame(seq);
        t_f += now_s() - t0;
        if (in_batch == 0)
            batch_first = seq;
        if (++in_batch < cfg.integrate)
            continue;

        /* F-step for the whole integration, parallel over antennas and blocks. */
        t0 = now_s();
#pragma omp parallel for collapse(2) schedule(static)
        for (int a = 0; a < cfg.n_ant; a++)
            for (int sp = 0; sp < n_spec_batch; sp++)
                fft_forward(&plan, spectra + 2 * ((size_t)a * ant_span + (size_t)sp * cfg.nchan));
        double t1 = now_s();
        xstep(spectra, cfg.n_ant, n_spec_batch, cfg.nchan, vis);
        double t2 = now_s();
        t_f += t1 - t0;
        t_x += t2 - t1;

        if (out) {
            rt_dump_header dh = {RT_DUMP_MAGIC, (uint32_t)n_bl, (uint32_t)cfg.nchan, (uint32_t)in_batch,
                                 (uint64_t)batch_first};
            fwrite(&dh, sizeof(dh), 1, out);
            fwrite(vis, sizeof(float), 2 * (size_t)n_bl * cfg.nchan, out);
        }
        frames += in_batch;
        in_batch = 0;
        dumps++;
    }
    double elapsed = frames ? now_s() - t_start : 0.0;

    for (int a = 0; a < cfg.n_ant; a++)
        pthread_join(th[a], NULL);
    uint64_t lost = 0;
    for (int a = 0; a < cfg.n_ant; a++)
        lost += rings[a].lost;

    double msps = elapsed > 0 ? frames * (double)cfg.frame / elapsed / 1e6 : 0.0;
    printf("corrd: n_ant=%d n_bl=%d nchan=%d frames=%lld dumps=%lld lost=%llu discarded=%lld\n", cfg.n_ant, n_bl,
           cfg.nchan, (long long)frames, (long long)dumps, (unsigned long long)lost, (long long)discarded);
    printf("corrd: elapsed %.3f s | per antenna %.2f MS/s | input %.1f MB/s | F-step %.3f s | X-step %.3f s\n",
           elapsed, msps, msps * 2 * cfg.n_ant, t_f, t_x);
    printf("corrd: real-time factor %.2f (vs %.2f MS/s per antenna); compute-only factor %.2f\n",
           msps / cfg.nominal_msps, cfg.nominal_msps,
           (t_f + t_x) > 0 ? frames * (double)cfg.frame / (t_f + t_x) / 1e6 / cfg.nominal_msps : 0.0);
    /* machine-readable summary line for benchmarks/bench_stream.sh */
    printf("CSV,%d,%d,%d,%lld,%llu,%.6f,%.6f,%.6f,%.4f\n", cfg.n_ant, cfg.nchan, cfg.frame, (long long)frames,
           (unsigned long long)lost, elapsed, t_f, t_x, msps);

    if (out)
        fclose(out);
    fft_plan_free(&plan);
    return 0;
}
