/*
 * rx_sim: simulate N radio receivers streaming 8-bit I/Q frames over TCP.
 *
 * One thread and one TCP connection per antenna (like separate receiver
 * boxes on campus).  Samples come from a file written by
 * benchmarks/stream_tools.py (looped if more frames are requested) or, without
 * --input, from a built-in generator: a common noise "sky" signal plus
 * independent receiver noise.
 *
 * With --rate R (Msamples/s per antenna) frames are paced on an absolute
 * schedule, like a real ADC.  When the correlator falls behind, TCP
 * back-pressure delays the sends; a real receiver would hold those samples in
 * its buffer (--buffer frames) and lose data only once the lag exceeds it.
 * Such frames are counted as overruns, and the maximum lag is reported.
 * --rate 0 sends as fast as back-pressure allows, which measures the
 * correlator's maximum sustainable throughput.
 *
 * Replaces legacy/server2.c and legacy/server2b.c.
 */
#define _GNU_SOURCE
#include "frame.h"

#include <arpa/inet.h>
#include <errno.h>
#include <getopt.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

typedef struct {
    const char *host;
    int port;
    int n_ant;
    int frame;            /* samples per frame */
    uint64_t n_frames;    /* frames per antenna to send */
    double rate_msps;     /* 0 = unlimited */
    int buffer_frames;    /* receiver buffer depth used to count overruns */
    const int8_t *data;   /* [n_ant][file_samples][2] or NULL */
    uint64_t file_samples;
} config;

/* All receivers start their sample clocks at the same instant, as real
 * receivers locked to a common (e.g. GPS-disciplined) reference would. */
static pthread_barrier_t start_barrier;
static uint64_t start_ns;

typedef struct {
    const config *cfg;
    int ant;
    uint64_t overruns;
    uint64_t sent;
    double max_lag_ms;
} worker;

static uint64_t now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + ts.tv_nsec;
}

static int send_all(int fd, const void *buf, size_t len)
{
    const char *p = buf;
    while (len > 0) {
        ssize_t n = send(fd, p, len, MSG_NOSIGNAL);
        if (n < 0) {
            if (errno == EINTR)
                continue;
            return -1;
        }
        p += n;
        len -= (size_t)n;
    }
    return 0;
}

/* xorshift64* -> approximately Gaussian integer (sum of 4 uniforms), rms ~ 0.58 * scale. */
static inline int noise(uint64_t *s, int scale)
{
    int acc = 0;
    for (int k = 0; k < 4; k++) {
        *s ^= *s >> 12;
        *s ^= *s << 25;
        *s ^= *s >> 27;
        acc += (int)((*s * 2685821657736338717ull) >> 56) - 128;
    }
    return acc * scale / 128;
}

static void fill_synthetic(int8_t *iq, int n, int ant, uint64_t seq)
{
    /* Common signal seeded by frame number only; noise seeded per antenna. */
    uint64_t s_sky = 0x9E3779B97F4A7C15ull ^ (seq * 0xBF58476D1CE4E5B9ull);
    uint64_t s_rx = (0xD1B54A32D192ED03ull * (ant + 1)) ^ (seq * 0x94D049BB133111EBull) ^ 1;
    for (int i = 0; i < 2 * n; i++) {
        int v = noise(&s_sky, 24) + noise(&s_rx, 24);
        iq[i] = (int8_t)(v > 127 ? 127 : v < -127 ? -127 : v);
    }
}

static int connect_to(const char *host, int port)
{
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0)
        return -1;
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    struct sockaddr_in addr = {.sin_family = AF_INET, .sin_port = htons(port), .sin_addr = {0}};
    if (inet_pton(AF_INET, host, &addr.sin_addr) != 1) {
        close(fd);
        return -1;
    }
    for (int attempt = 0; attempt < 50; attempt++) { /* corrd may still be starting */
        if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) == 0)
            return fd;
        usleep(100000);
    }
    close(fd);
    return -1;
}

static void *run_antenna(void *arg)
{
    worker *w = arg;
    const config *c = w->cfg;
    int fd = connect_to(c->host, c->port);
    if (fd < 0)
        fprintf(stderr, "rx_sim[%d]: cannot connect to %s:%d\n", w->ant, c->host, c->port);
    if (pthread_barrier_wait(&start_barrier) == PTHREAD_BARRIER_SERIAL_THREAD)
        start_ns = now_ns() + 5000000ull; /* common epoch 5 ms after the last connection */
    pthread_barrier_wait(&start_barrier);
    if (fd < 0)
        return NULL;

    size_t payload = 2 * (size_t)c->frame;
    char *buf = malloc(sizeof(rt_frame_header) + payload);
    rt_frame_header *h = (rt_frame_header *)buf;
    int8_t *iq = (int8_t *)(buf + sizeof(rt_frame_header));
    double period_ns = c->rate_msps > 0 ? c->frame * 1e3 / c->rate_msps : 0.0;
    const uint64_t t0 = start_ns;

    for (uint64_t seq = 0; seq < c->n_frames; seq++) {
        if (c->data) {
            uint64_t off = (seq * c->frame) % c->file_samples;
            const int8_t *src = c->data + 2 * ((uint64_t)w->ant * c->file_samples);
            for (int i = 0; i < c->frame; i++) {
                uint64_t k = (off + i) % c->file_samples;
                iq[2 * i] = src[2 * k];
                iq[2 * i + 1] = src[2 * k + 1];
            }
        } else {
            fill_synthetic(iq, c->frame, w->ant, seq);
        }

        if (period_ns > 0) {
            uint64_t due = t0 + (uint64_t)(seq * period_ns);
            uint64_t t = now_ns();
            if (t < due) {
                struct timespec ts = {(time_t)(due / 1000000000ull), (long)(due % 1000000000ull)};
                clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, NULL);
            } else {
                double lag = (double)(t - due);
                if (lag * 1e-6 > w->max_lag_ms)
                    w->max_lag_ms = lag * 1e-6;
                if (lag > c->buffer_frames * period_ns)
                    w->overruns++;
            }
        }

        *h = (rt_frame_header){RT_FRAME_MAGIC, (uint16_t)w->ant, (uint16_t)c->n_ant, seq, now_ns(),
                               (uint32_t)c->frame, 0};
        if (send_all(fd, buf, sizeof(rt_frame_header) + payload) != 0) {
            perror("rx_sim send");
            break;
        }
        w->sent++;
    }
    free(buf);
    close(fd);
    return NULL;
}

static int8_t *load_file(const char *path, int *n_ant, uint64_t *n_samples)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        perror(path);
        return NULL;
    }
    uint32_t magic, na;
    uint64_t ns;
    if (fread(&magic, 4, 1, f) != 1 || fread(&na, 4, 1, f) != 1 || fread(&ns, 8, 1, f) != 1 ||
        magic != RT_FILE_MAGIC) {
        fprintf(stderr, "%s: not an rx_sim input file\n", path);
        fclose(f);
        return NULL;
    }
    size_t bytes = (size_t)na * ns * 2;
    int8_t *data = malloc(bytes);
    if (!data || fread(data, 1, bytes, f) != bytes) {
        fprintf(stderr, "%s: truncated\n", path);
        free(data);
        fclose(f);
        return NULL;
    }
    fclose(f);
    *n_ant = (int)na;
    *n_samples = ns;
    return data;
}

static void usage(const char *prog)
{
    fprintf(stderr,
            "usage: %s [--host IP] [--port P] [--n-ant N] [--frame SAMPLES] [--frames F]\n"
            "          [--rate MSPS_PER_ANT] [--buffer FRAMES] [--input FILE]\n",
            prog);
}

int main(int argc, char **argv)
{
    config c = {.host = "127.0.0.1", .port = 5000, .n_ant = 4, .frame = 4096, .n_frames = 1000, .buffer_frames = 32};
    const char *input = NULL;
    static struct option opts[] = {{"host", required_argument, 0, 'h'},   {"port", required_argument, 0, 'p'},
                                   {"n-ant", required_argument, 0, 'n'},  {"frame", required_argument, 0, 'f'},
                                   {"frames", required_argument, 0, 'F'}, {"rate", required_argument, 0, 'r'},
                                   {"input", required_argument, 0, 'i'},  {"buffer", required_argument, 0, 'b'},
                                   {0, 0, 0, 0}};
    int o;
    while ((o = getopt_long(argc, argv, "", opts, NULL)) != -1) {
        switch (o) {
        case 'h': c.host = optarg; break;
        case 'p': c.port = atoi(optarg); break;
        case 'n': c.n_ant = atoi(optarg); break;
        case 'f': c.frame = atoi(optarg); break;
        case 'F': c.n_frames = strtoull(optarg, NULL, 10); break;
        case 'r': c.rate_msps = atof(optarg); break;
        case 'i': input = optarg; break;
        case 'b': c.buffer_frames = atoi(optarg); break;
        default: usage(argv[0]); return 2;
        }
    }
    if (input) {
        int na;
        c.data = load_file(input, &na, &c.file_samples);
        if (!c.data)
            return 1;
        c.n_ant = na;
    }
    if (c.n_ant < 2 || c.frame <= 0) {
        usage(argv[0]);
        return 2;
    }

    worker *w = calloc(c.n_ant, sizeof(worker));
    pthread_t *th = calloc(c.n_ant, sizeof(pthread_t));
    pthread_barrier_init(&start_barrier, NULL, (unsigned)c.n_ant);
    uint64_t t0 = now_ns();
    for (int a = 0; a < c.n_ant; a++) {
        w[a] = (worker){&c, a, 0, 0, 0.0};
        pthread_create(&th[a], NULL, run_antenna, &w[a]);
    }
    uint64_t overruns = 0, sent = 0;
    double max_lag = 0.0;
    for (int a = 0; a < c.n_ant; a++) {
        pthread_join(th[a], NULL);
        overruns += w[a].overruns;
        sent += w[a].sent;
        if (w[a].max_lag_ms > max_lag)
            max_lag = w[a].max_lag_ms;
    }
    double secs = (now_ns() - t0) * 1e-9;
    printf("rx_sim: %d antennas, %llu frames sent, %.3f s, %.2f MS/s per antenna\n", c.n_ant,
           (unsigned long long)sent, secs, sent * (double)c.frame / c.n_ant / secs / 1e6);
    if (c.rate_msps > 0)
        printf("rx_sim: max lag %.2f ms, %llu overruns (lag > %d-frame receiver buffer)\n", max_lag,
               (unsigned long long)overruns, c.buffer_frames);
    free((void *)c.data);
    free(w);
    free(th);
    return sent == (uint64_t)c.n_ant * c.n_frames ? 0 : 1;
}
