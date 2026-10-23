/*
 * Iterative radix-2 decimation-in-time FFT.  Replaces the recursive version in
 * legacy/fft_basic.c (which allocated on the stack at every level) and the
 * incorrect AVX attempt in legacy/fft_avx.c.  Twiddles and the bit-reversal
 * permutation are computed once per plan.
 */
#include "fft.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

int fft_plan_init(fft_plan *p, int n)
{
    memset(p, 0, sizeof(*p));
    if (n < 2 || (n & (n - 1)) != 0)
        return -1;

    p->n = n;
    while ((1 << p->log2n) < n)
        p->log2n++;

    p->bitrev = malloc(sizeof(int) * n);
    p->twiddle = malloc(sizeof(float) * n); /* n/2 complex */
    if (!p->bitrev || !p->twiddle) {
        fft_plan_free(p);
        return -1;
    }

    for (int i = 0; i < n; i++) {
        int r = 0;
        for (int b = 0; b < p->log2n; b++)
            r |= ((i >> b) & 1) << (p->log2n - 1 - b);
        p->bitrev[i] = r;
    }
    for (int k = 0; k < n / 2; k++) {
        double ang = -2.0 * M_PI * k / n;
        p->twiddle[2 * k] = (float)cos(ang);
        p->twiddle[2 * k + 1] = (float)sin(ang);
    }
    return 0;
}

void fft_plan_free(fft_plan *p)
{
    free(p->bitrev);
    free(p->twiddle);
    memset(p, 0, sizeof(*p));
}

void fft_forward(const fft_plan *p, float *x)
{
    const int n = p->n;

    for (int i = 0; i < n; i++) {
        int j = p->bitrev[i];
        if (j > i) {
            float tr = x[2 * i], ti = x[2 * i + 1];
            x[2 * i] = x[2 * j];
            x[2 * i + 1] = x[2 * j + 1];
            x[2 * j] = tr;
            x[2 * j + 1] = ti;
        }
    }

    for (int len = 2; len <= n; len <<= 1) {
        const int half = len >> 1;
        const int step = n / len; /* twiddle stride */
        for (int start = 0; start < n; start += len) {
            for (int k = 0; k < half; k++) {
                const float wr = p->twiddle[2 * k * step], wi = p->twiddle[2 * k * step + 1];
                float *u = x + 2 * (start + k);
                float *v = x + 2 * (start + k + half);
                const float vr = v[0] * wr - v[1] * wi;
                const float vi = v[0] * wi + v[1] * wr;
                v[0] = u[0] - vr;
                v[1] = u[1] - vi;
                u[0] += vr;
                u[1] += vi;
            }
        }
    }
}

int fft_forward_batch(float *data, int n, int count)
{
    fft_plan p;
    if (fft_plan_init(&p, n) != 0)
        return -1;
    for (int i = 0; i < count; i++)
        fft_forward(&p, data + 2 * (size_t)i * n);
    fft_plan_free(&p);
    return 0;
}
