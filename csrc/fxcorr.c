/*
 * X-step kernels: scalar reference, AVX2+FMA, and AVX2+FMA with OpenMP.
 * See fxcorr.h for the data layout.
 */
#include "fxcorr.h"

#include <immintrin.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

int fx_n_baselines(int n_ant) { return n_ant * (n_ant - 1) / 2; }

int fx_cpu_has_avx2(void)
{
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
}

int fx_max_threads(void)
{
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

void fx_set_threads(int n)
{
#ifdef _OPENMP
    if (n > 0)
        omp_set_num_threads(n);
#else
    (void)n;
#endif
}

/* ------------------------------------------------------------------------ */
/* Scalar reference.  Auto-vectorisation is disabled so it really is scalar. */

__attribute__((optimize("no-tree-vectorize")))
static void xstep_one_scalar(const float *a, const float *b, int n_spec, int nchan,
                             int c_begin, float *out)
{
    const float norm = 1.0f / (float)n_spec;
    for (int c = c_begin; c < nchan; c++) {
        float re = 0.0f, im = 0.0f;
        for (int t = 0; t < n_spec; t++) {
            const float *pa = a + 2 * ((size_t)t * nchan + c);
            const float *pb = b + 2 * ((size_t)t * nchan + c);
            /* a * conj(b) */
            re += pa[0] * pb[0] + pa[1] * pb[1];
            im += pa[1] * pb[0] - pa[0] * pb[1];
        }
        out[2 * c] = re * norm;
        out[2 * c + 1] = im * norm;
    }
}

void fx_xstep_scalar(const float *spectra, int n_ant, int n_spec, int nchan, float *vis)
{
    const size_t ant_stride = 2 * (size_t)n_spec * nchan;
    int b = 0;
    for (int i = 0; i < n_ant; i++)
        for (int j = i + 1; j < n_ant; j++, b++)
            xstep_one_scalar(spectra + i * ant_stride, spectra + j * ant_stride, n_spec, nchan,
                             0, vis + 2 * (size_t)b * nchan);
}

/* ------------------------------------------------------------------------ */
/* AVX2 + FMA, cache-blocked.
 *
 * Arithmetic.  A __m256 holds 4 interleaved complex numbers [ar0 ai0 ar1 ai1 ...].
 * Rather than shuffling every product, accumulate over time
 *     P += a * b          -> [ar*br, ai*bi, ...]
 *     Q += a * swap(b)    -> [ar*bi, ai*br, ...]
 * and combine once per channel block:
 *     re = P_even + P_odd,  im = Q_odd - Q_even   (= Re/Im of a * conj(b)).
 *
 * Memory.  Walking the spectra baseline by baseline strides nchan * 8 bytes
 * per time step and re-reads every antenna N-1 times from DRAM, which makes
 * the X-step bandwidth-bound.  Instead the channels are processed in blocks of
 * BLK = 8: each block is first packed into a contiguous buffer
 * buf[ant][t][8 complex] (one 64-byte cache line per antenna and time step),
 * which stays in L1/L2 while all N(N-1)/2 baselines are formed from it.  The
 * inner loop is register-tiled 1x2 (antenna i against j and j+1) so each load of
 * antenna i feeds two baselines.  OpenMP parallelises over channel blocks, each
 * thread with its own pack buffer, so threads share no output and no DRAM
 * traffic beyond the single packing pass.
 */

#define BLK 8 /* complex channels per block = 2 __m256 */

__attribute__((target("avx2,fma")))
static inline __m256 combine(__m256 p, __m256 q, __m256 norm)
{
    __m256 re = _mm256_add_ps(p, _mm256_permute_ps(p, 0xB1)); /* re in both slots */
    __m256 im = _mm256_sub_ps(q, _mm256_permute_ps(q, 0xB1)); /* im in odd slots */
    return _mm256_mul_ps(_mm256_blend_ps(re, im, 0xAA), norm);
}

/* Copy channels [c0, c0 + BLK) of every antenna into buf[ant][t][2 * BLK]. */
static void pack_block(const float *spectra, int n_ant, int n_spec, int nchan, int c0, float *buf)
{
    for (int a = 0; a < n_ant; a++)
        for (int t = 0; t < n_spec; t++)
            memcpy(buf + 2 * BLK * ((size_t)a * n_spec + t),
                   spectra + 2 * (((size_t)a * n_spec + t) * nchan + c0), sizeof(float) * 2 * BLK);
}

/* All baselines for one packed channel block; results go to vis[b][c0 .. c0 + BLK). */
__attribute__((target("avx2,fma")))
static void xstep_block_avx2(const float *buf, int n_ant, int n_spec, int nchan, int c0, float *vis)
{
    const __m256 norm = _mm256_set1_ps(1.0f / (float)n_spec);
    const size_t ant = 2 * BLK * (size_t)n_spec; /* floats per antenna in buf */
    int b = 0;                                    /* triu_indices order: (0,1),(0,2),...,(1,2),... */
    for (int i = 0; i < n_ant; i++) {
        const float *pa = buf + i * ant;
        int j = i + 1;
        for (; j + 1 < n_ant; j += 2, b += 2) { /* two baselines per pass over antenna i */
            const float *pb = buf + j * ant, *pc = buf + (j + 1) * ant;
            __m256 p0 = _mm256_setzero_ps(), p1 = p0, q0 = p0, q1 = p0;
            __m256 r0 = p0, r1 = p0, s0 = p0, s1 = p0;
            for (int t = 0; t < n_spec; t++) {
                const size_t o = 2 * BLK * (size_t)t;
                __m256 a0 = _mm256_load_ps(pa + o), a1 = _mm256_load_ps(pa + o + 8);
                __m256 b0 = _mm256_load_ps(pb + o), b1 = _mm256_load_ps(pb + o + 8);
                __m256 c0v = _mm256_load_ps(pc + o), c1v = _mm256_load_ps(pc + o + 8);
                p0 = _mm256_fmadd_ps(a0, b0, p0);
                p1 = _mm256_fmadd_ps(a1, b1, p1);
                q0 = _mm256_fmadd_ps(a0, _mm256_permute_ps(b0, 0xB1), q0);
                q1 = _mm256_fmadd_ps(a1, _mm256_permute_ps(b1, 0xB1), q1);
                r0 = _mm256_fmadd_ps(a0, c0v, r0);
                r1 = _mm256_fmadd_ps(a1, c1v, r1);
                s0 = _mm256_fmadd_ps(a0, _mm256_permute_ps(c0v, 0xB1), s0);
                s1 = _mm256_fmadd_ps(a1, _mm256_permute_ps(c1v, 0xB1), s1);
            }
            float *out = vis + 2 * ((size_t)b * nchan + c0);
            _mm256_storeu_ps(out, combine(p0, q0, norm));
            _mm256_storeu_ps(out + 8, combine(p1, q1, norm));
            out += 2 * (size_t)nchan;
            _mm256_storeu_ps(out, combine(r0, s0, norm));
            _mm256_storeu_ps(out + 8, combine(r1, s1, norm));
        }
        for (; j < n_ant; j++, b++) { /* odd remainder */
            const float *pb = buf + j * ant;
            __m256 p0 = _mm256_setzero_ps(), p1 = p0, q0 = p0, q1 = p0;
            for (int t = 0; t < n_spec; t++) {
                const size_t o = 2 * BLK * (size_t)t;
                __m256 a0 = _mm256_load_ps(pa + o), a1 = _mm256_load_ps(pa + o + 8);
                __m256 b0 = _mm256_load_ps(pb + o), b1 = _mm256_load_ps(pb + o + 8);
                p0 = _mm256_fmadd_ps(a0, b0, p0);
                p1 = _mm256_fmadd_ps(a1, b1, p1);
                q0 = _mm256_fmadd_ps(a0, _mm256_permute_ps(b0, 0xB1), q0);
                q1 = _mm256_fmadd_ps(a1, _mm256_permute_ps(b1, 0xB1), q1);
            }
            float *out = vis + 2 * ((size_t)b * nchan + c0);
            _mm256_storeu_ps(out, combine(p0, q0, norm));
            _mm256_storeu_ps(out + 8, combine(p1, q1, norm));
        }
    }
}

/* Channels beyond the last full block, all baselines, scalar. */
static void xstep_tail(const float *spectra, int n_ant, int n_spec, int nchan, int c_begin, float *vis)
{
    const size_t ant_stride = 2 * (size_t)n_spec * nchan;
    int b = 0;
    for (int i = 0; i < n_ant; i++)
        for (int j = i + 1; j < n_ant; j++, b++)
            xstep_one_scalar(spectra + i * ant_stride, spectra + j * ant_stride, n_spec, nchan, c_begin,
                             vis + 2 * (size_t)b * nchan);
}

static float *alloc_pack(int n_ant, int n_spec)
{
    size_t bytes = sizeof(float) * 2 * BLK * (size_t)n_ant * n_spec;
    return aligned_alloc(64, (bytes + 63) / 64 * 64);
}

void fx_xstep_avx2(const float *spectra, int n_ant, int n_spec, int nchan, float *vis)
{
    const int n_blocks = nchan / BLK;
    float *buf = alloc_pack(n_ant, n_spec);
    for (int k = 0; k < n_blocks; k++) {
        pack_block(spectra, n_ant, n_spec, nchan, k * BLK, buf);
        xstep_block_avx2(buf, n_ant, n_spec, nchan, k * BLK, vis);
    }
    free(buf);
    if (n_blocks * BLK < nchan)
        xstep_tail(spectra, n_ant, n_spec, nchan, n_blocks * BLK, vis);
}

void fx_xstep_avx2_omp(const float *spectra, int n_ant, int n_spec, int nchan, float *vis)
{
    const int n_blocks = nchan / BLK;
#pragma omp parallel
    {
        float *buf = alloc_pack(n_ant, n_spec);
#pragma omp for schedule(static)
        for (int k = 0; k < n_blocks; k++) {
            pack_block(spectra, n_ant, n_spec, nchan, k * BLK, buf);
            xstep_block_avx2(buf, n_ant, n_spec, nchan, k * BLK, vis);
        }
        free(buf);
    }
    if (n_blocks * BLK < nchan)
        xstep_tail(spectra, n_ant, n_spec, nchan, n_blocks * BLK, vis);
}
