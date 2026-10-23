#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1024
#define M_PI 3.1415


void fft(double complex *X, int n) {
    if (n <= 1) return;

    double complex X_even[n/2];
    double complex X_odd[n/2];

    for (int i = 0; i < n/2; i++) {
        X_even[i] = X[i*2];
        X_odd[i] = X[i*2 + 1];
    }

    fft(X_even, n/2);
    fft(X_odd, n/2);

    for (int i = 0; i < n/2; i++) {
        double complex t = cexp(-2.0*I*M_PI*i/n) * X_odd[i];
        X[i] = X_even[i] + t;
        X[i + n/2] = X_even[i] - t;
    }
}

void ifft(double complex *X, int n) {
    if (n <= 1) return;

    for (int i = 0; i < n; i++) {
        X[i] = conj(X[i]);
    }

    fft(X, n);

    for (int i = 0; i < n; i++) {
        X[i] = conj(X[i]) / n;
    }
}

int main() {
    double complex x[N];
    double complex x_orig[N];

    // Generate a sine wave signal
    for (int i = 0; i < N; i++) {
        x[i] = cexp(I * 2 * M_PI * i / N);
        x_orig[i] = x[i]; // Save the original signal
    }

    // Perform FFT
    fft(x, N);

    // Perform IFFT
    ifft(x, N);

    // Compare the original and the IFFT results
    for (int i = 0; i < N; i++) {
        printf("Original: %f + %fi, IFFT: %f + %fi\n",
               creal(x_orig[i]), cimag(x_orig[i]),
               creal(x[i]), cimag(x[i]));
    }

    return 0;
}
