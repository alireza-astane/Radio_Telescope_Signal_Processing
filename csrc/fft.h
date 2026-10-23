/* Minimal iterative radix-2 complex FFT (single precision) for the streaming daemon. */
#ifndef RT_FFT_H
#define RT_FFT_H

typedef struct {
    int n;
    int log2n;
    int *bitrev;
    float *twiddle; /* n/2 interleaved complex exp(-2 pi i k / n) */
} fft_plan;

/* Returns 0 on success, -1 if n is not a power of two or allocation fails. */
int fft_plan_init(fft_plan *p, int n);
void fft_plan_free(fft_plan *p);

/* In-place forward FFT of n interleaved complex floats. */
void fft_forward(const fft_plan *p, float *data);

/* Convenience: plan + transform `count` consecutive length-n blocks in place. */
int fft_forward_batch(float *data, int n, int count);

#endif
