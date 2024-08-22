#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fftw3.h> // Include FFTW3 library

// Function to perform Hilbert Transform using FFTW3
void hilbert_transform(double *signal, int length, double *hilbert_signal) {
    // Allocate memory for complex data
    fftw_complex *complex_signal = (fftw_complex*)malloc(sizeof(fftw_complex) * length);
    fftw_complex *complex_hilbert = (fftw_complex*)malloc(sizeof(fftw_complex) * length);

    // Copy real signal to complex data
    for (int i = 0; i < length; i++) {
        complex_signal[i][0] = signal[i];
        complex_signal[i][1] = 0.0;
    }

    // Perform FFT
    fftw_plan plan_forward = fftw_plan_dft_1d(length, complex_signal, complex_hilbert, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(plan_forward);

    // Apply Hilbert transform by multiplying imaginary components by -1
    for (int i = 1; i < length / 2; i++) {
        complex_hilbert[i][1] *= -1.0;
    }
    for (int i = length / 2 + 1; i < length; i++) {
        complex_hilbert[i][1] *= -1.0;
    }

    // Perform inverse FFT
    fftw_plan plan_backward = fftw_plan_dft_1d(length, complex_hilbert, complex_signal, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(plan_backward);

    // Extract the imaginary component as the Hilbert Transform
    for (int i = 0; i < length; i++) {
        hilbert_signal[i] = complex_signal[i][1] / length;
    }

    // Free memory
    fftw_destroy_plan(plan_forward);
    fftw_destroy_plan(plan_backward)

;
    fftw_free(complex_signal);
    fftw_free(complex_hilbert);
}

// Function to compute the phase difference between two signals
double phase_difference(double *signal1, double *signal2, int length) {
    double phase_diff = 0.0;
    for (int i = 0; i < length; i++) {
        phase_diff += atan2(signal2[i], signal1[i]) - atan2(signal1[i], signal2[i]);
    }
    return phase_diff / length; // Average phase difference
}

int main() {
    int length;

    // Input the length of the arrays
    printf("Enter the length of the data arrays: ");
    scanf("%d", &length);

    // Allocate memory for the data arrays
    double *signal1 = (double*)malloc(sizeof(double) * length);
    double *signal2 = (double*)malloc(sizeof(double) * length);
    double *hilbert1 = (double*)malloc(sizeof(double) * length);
    double *hilbert2 = (double*)malloc(sizeof(double) * length);

    // Input the data for the arrays
    printf("Enter the data for signal 1:\n");
    for (int i = 0; i < length; i++) {
        scanf("%lf", &signal1[i]);
    }
    printf("Enter the data for signal 2:\n");
    for (int i = 0; i < length; i++) {
        scanf("%lf", &signal2[i]);
    }

    // Perform Hilbert Transform
    hilbert_transform(signal1, length, hilbert1);
    hilbert_transform(signal2, length, hilbert2);

    // Compute phase difference
    double phase_diff = phase_difference(hilbert1, hilbert2, length);

    // Print the phase difference
    printf("Phase difference: %lf\n", phase_diff);

    // Free memory
    free(signal1);
    free(signal2);
    free(hilbert1);
    free(hilbert2);

    return 0;
}


