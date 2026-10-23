#include <immintrin.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#define N 256 // Number of points in FFT (must be a power of 2)
#define M_PI 3.1415

// Function to perform FFT using AVX data input
void fft_avx(__m256d* real, __m256d* imag, int n) {
    int logn = log2(n);
    __m256d temp1, temp2, temp3, temp4;
    __m256d twiddle_re, twiddle_im;
    
    for (int i = 0; i < logn; ++i) {
        int m = 1 << (i + 1);
        int half_m = m >> 1;

        double theta = -2.0 * M_PI / m;

        for (int j = 0; j < half_m; j += 4) {
            __m256d cos_theta = _mm256_set1_pd(cos(theta * j));
            __m256d sin_theta = _mm256_set1_pd(sin(theta * j));
            
            twiddle_re = cos_theta;
            twiddle_im = sin_theta;

            for (int k = j; k < n; k += m) {
                int l = k + half_m;

                temp1 = _mm256_loadu_pd((double *)&real[k]);
                temp2 = _mm256_loadu_pd((double *)&real[l]);
                temp3 = _mm256_loadu_pd((double *)&imag[k]);
                temp4 = _mm256_loadu_pd((double *)&imag[l]);

                // Calculate real part
                __m256d real_twiddle = _mm256_sub_pd(_mm256_mul_pd(temp2, twiddle_re), _mm256_mul_pd(temp4, twiddle_im));
                __m256d imag_twiddle = _mm256_add_pd(_mm256_mul_pd(temp4, twiddle_re), _mm256_mul_pd(temp2, twiddle_im));

                // Store the results
                _mm256_storeu_pd((double *)&real[k], _mm256_add_pd(temp1, real_twiddle));
                _mm256_storeu_pd((double *)&real[l], _mm256_sub_pd(temp1, real_twiddle));

                _mm256_storeu_pd((double *)&imag[k], _mm256_add_pd(temp3, imag_twiddle));
                _mm256_storeu_pd((double *)&imag[l], _mm256_sub_pd(temp3, imag_twiddle));
            }
        }
    }
}


// Function to perform IFFT using AVX data input
void ifft_avx(__m256d* real, __m256d* imag, int n) {
    int logn = log2(n);
    __m256d temp1, temp2, temp3, temp4;
    __m256d twiddle_re, twiddle_im;
    
    for (int i = 0; i < logn; ++i) {
        int m = 1 << (i + 1);
        int half_m = m >> 1;

        double theta = 2.0 * M_PI / m;

        for (int j = 0; j < half_m; j += 4) {
            __m256d cos_theta = _mm256_set1_pd(cos(theta * j));
            __m256d sin_theta = _mm256_set1_pd(sin(theta * j));
            
            twiddle_re = cos_theta;
            twiddle_im = sin_theta;

            for (int k = j; k < n; k += m) {
                int l = k + half_m;

                temp1 = _mm256_loadu_pd((double *)&real[k]);
                temp2 = _mm256_loadu_pd((double *)&real[l]);
                temp3 = _mm256_loadu_pd((double *)&imag[k]);
                temp4 = _mm256_loadu_pd((double *)&imag[l]);

                // Calculate real part
                __m256d real_twiddle = _mm256_sub_pd(_mm256_mul_pd(temp2, twiddle_re), _mm256_mul_pd(temp4, twiddle_im));
                __m256d imag_twiddle = _mm256_add_pd(_mm256_mul_pd(temp4, twiddle_re), _mm256_mul_pd(temp2, twiddle_im));

                // Store the results
                _mm256_storeu_pd((double *)&real[k], _mm256_add_pd(temp1, real_twiddle));
                _mm256_storeu_pd((double *)&real[l], _mm256_sub_pd(temp1, real_twiddle));

                _mm256_storeu_pd((double *)&imag[k], _mm256_add_pd(temp3, imag_twiddle));
                _mm256_storeu_pd((double *)&imag[l], _mm256_sub_pd(temp3, imag_twiddle));
            }
        }
    }

    // Normalize the IFFT result
    __m256d norm = _mm256_set1_pd(1.0 / n);
    for (int i = 0; i < n; i += 4) {
        temp1 = _mm256_loadu_pd((double *)&real[i]);
        temp2 = _mm256_loadu_pd((double *)&imag[i]);

        temp1 = _mm256_mul_pd(temp1, norm);
        temp2 = _mm256_mul_pd(temp2, norm);

        _mm256_storeu_pd((double *)&real[i], temp1);
        _mm256_storeu_pd((double *)&imag[i], temp2);
    }
}
#define TOLERANCE 1e-12 // Tolerance for floating-point comparison

// Function to print complex array
void print_complex_array(__m256d* real, __m256d* imag, int n) {
    for (int i = 0; i < n; i++) {
        double real_part[4], imag_part[4];
        _mm256_storeu_pd(real_part, real[i/4]);
        _mm256_storeu_pd(imag_part, imag[i/4]);
        printf("%d: %f + %fi\n", i, real_part[i % 4], imag_part[i % 4]);
    }
}

// Function to compare initial and final arrays with tolerance
void compare_arrays(__m256d* real1, __m256d* imag1, __m256d* real2, __m256d* imag2, int n) {
    for (int i = 0; i < n; i++) {
        double real1_part[4], imag1_part[4];
        double real2_part[4], imag2_part[4];

        _mm256_storeu_pd(real1_part, real1[i/4]);
        _mm256_storeu_pd(imag1_part, imag1[i/4]);

        _mm256_storeu_pd(real2_part, real2[i/4]);
        _mm256_storeu_pd(imag2_part, imag2[i/4]);

        if (fabs(real1_part[i % 4] - real2_part[i % 4]) > TOLERANCE || fabs(imag1_part[i % 4] - imag2_part[i % 4]) > TOLERANCE) {
            printf("Difference found at index %d: %f + %fi vs %f + %fi\n", i, real1_part[i % 4], imag1_part[i % 4], real2_part[i % 4], imag2_part[i % 4]);
        }
    }
}

int main() {
    int n = N;

    // Allocate memory for real and imaginary parts of the signal
    __m256d* real = (__m256d*)aligned_alloc(32, sizeof(__m256d) * n / 4);
    __m256d* imag = (__m256d*)aligned_alloc(32, sizeof(__m256d) * n / 4);
    __m256d* original_real = (__m256d*)aligned_alloc(32, sizeof(__m256d) * n / 4);
    __m256d* original_imag = (__m256d*)aligned_alloc(32, sizeof(__m256d) * n / 4);

    // Initialize the sine wave array
    for (int i = 0; i < n / 4; ++i) {
        real[i] = _mm256_set_pd(sin(2 * M_PI * (4*i+3) / n), 
                                sin(2 * M_PI * (4*i+2) / n), 
                                sin(2 * M_PI * (4*i+1) / n), 
                                sin(2 * M_PI * 4*i / n));
        imag[i] = _mm256_setzero_pd();
        
        original_real[i] = real[i]; // Store the initial array
        original_imag[i] = imag[i];
    }

    // Print the initial array
    printf("Initial Array:\n");
    print_complex_array(real, imag, n);

    // Perform FFT
    fft_avx(real, imag, n);

    // Perform IFFT
    ifft_avx(real, imag, n);

    // Print the resulting array after IFFT
    printf("\nArray after FFT and IFFT:\n");
    print_complex_array(real, imag, n);

    // Compare initial and final arrays
    printf("\nComparison of Initial and Final Arrays:\n");
    compare_arrays(original_real, original_imag, real, imag, n);

    // Cleanup
    free(real);
    free(imag);
    free(original_real);
    free(original_imag);

    return 0;
}