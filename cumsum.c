#include <stdio.h>
#include <immintrin.h>  // Header for AVX



typedef __m256i v8i;

v8i prefix(v8i x) {
    // x = 1, 2, 3, 4, 5, 6, 7, 8
    x = _mm256_add_epi32(x, _mm256_slli_si256(x, 4));
    x = _mm256_add_epi32(x, _mm256_slli_si256(x, 8));
    x = _mm256_add_epi32(x, _mm256_slli_si256(x, 16)); // <- this does nothing
    // x = 1, 3, 6, 10, 5, 11, 18, 26
    return x;
}

void avx_cumsum(double *arr, double *cumsum, int n) {
    if (n == 0) return;

    __m256i vec_sum = _mm256_setzero_pd();  // Initialize a vector to accumulate the partial sums

    int i;
    for (i = 0; i <= n - 4; i += 4) {
        __m256i vec = _mm256_loadu_pd(&arr[i]); // Load 4 elements from the input array
        vec_sum = _mm256_add_pd(vec_sum, vec);  // Add them to the partial sum vector

        // Store intermediate cumulative results in a temporary array
        _mm256_storeu_pd(&cumsum[i], vec_sum);
    }

    // Process remaining elements
    for (; i < n; i++) {
        cumsum[i] = cumsum[i - 1] + arr[i];
    }

    // Second pass to fix the cumulative sums
    for (i = 4; i < n; i++) {
        cumsum[i] += cumsum[i - 4];
    }
}

int main() {
    double arr[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    int n = sizeof(arr) / sizeof(arr[0]);
    double cumsum[n];

    avx_cumsum(arr, cumsum, n);

    printf("Cumulative sum:\n");
    for (int i = 0; i < n; i++) {
        printf("%f ", cumsum[i]);
    }
    printf("\n");

    return 0;
}
