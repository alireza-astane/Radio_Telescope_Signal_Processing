#include <stdio.h>
#include <stdlib.h>
#include <fftw3.h>
#include <math.h>

#define N 1024 // Number of points in FFT (must be a power of 2)
// Function to compute the Hilbert transform
void hilbert_transform(double* in, double* out, int n) {
    fftw_complex *fft_in, *fft_out;
    fftw_plan p_forward, p_backward;

    fft_in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n);
    fft_out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n);

    p_forward = fftw_plan_dft_r2c_1d(n, in, fft_in, FFTW_ESTIMATE);
    p_backward = fftw_plan_dft_c2r_1d(n, fft_out, out, FFTW_ESTIMATE);

    fftw_execute(p_forward);

    int half_n = n / 2;
    for (int i = 0; i < half_n; ++i) {
        double freq = (i < half_n) ? 1.0 : -1.0;
        fft_out[i][0] = fft_in[i][0] * freq;
        fft_out[i][1] = fft_in[i][1] * freq;
    }
    fft_out[half_n][0] = fft_in[half_n][0];
    fft_out[half_n][1] = fft_in[half_n][1];

    fftw_execute(p_backward);

    for (int i = 0; i < n; ++i) {
        out[i] /= n;
    }

    fftw_destroy_plan(p_forward);
    fftw_destroy_plan(p_backward);
    fftw_free(fft_in);
    fftw_free(fft_out);
}

// Function to compute the phase of a complex signal
void compute_phases(double* real, double* imag, double* phases, int n) {
    for (int i = 0; i < n; ++i) {
        phases[i] = atan2(imag[i], real[i]);
    }
}

// Function to unwrap phase angles
void unwrap_phases(double* phases, double* unwrapped, int n) {
    unwrapped[0] = phases[0];
    for (int i = 1; i < n; ++i) {
        double delta = phases[i] - unwrapped[i - 1];
        if (delta > M_PI) delta -= 2 * M_PI;
        if (delta < -M_PI) delta += 2 * M_PI;
        unwrapped[i] = unwrapped[i - 1] + delta;
    }
}

// Function to compute phase differences
void compute_phase_differences(double* phase1, double* phase2, double* differences, int n) {
    for (int i = 0; i < n; ++i) {
        differences[i] = phase1[i] - phase2[i];
        if (differences[i] > M_PI) differences[i] -= 2 * M_PI;
        if (differences[i] < -M_PI) differences[i] += 2 * M_PI;
    }
}

// Function to compute mean of an array
double mean(double* array, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += array[i];
    }
    return sum / n;
}

int main() {
    double time[N];
    double signal1[N], signal2[N];
    double hilbert1[N], hilbert2[N];
    double real1[N], imag1[N];
    double real2[N], imag2[N];
    double phases1[N], phases2[N];
    double unwrapped1[N], unwrapped2[N];
    double phase_diffs[N];
    
    // Generate time vector and two sine signals
    double frequency1 = 100000;
    double frequency2 = 100000;
    double phase1 = 0.0;
    double phase2 = M_PI / 2; // 45 degrees phase difference
    for (int i = 0; i < N; ++i) {
        time[i] = (double)i / N;
        signal1[i] = sin(2 * M_PI * frequency1 * time[i] + phase1);
        signal2[i] = sin(2 * M_PI * frequency2 * time[i] + phase2);
    }

    // Perform Hilbert transform
    hilbert_transform(signal1, hilbert1, N);
    hilbert_transform(signal2, hilbert2, N);

    // Compute phases
    compute_phases(signal1, hilbert1, phases1, N);
    compute_phases(signal2, hilbert2, phases2, N);

    // Unwrap phases
    unwrap_phases(phases1, unwrapped1, N);
    unwrap_phases(phases2, unwrapped2, N);

    // Compute phase differences
    compute_phase_differences(unwrapped1, unwrapped2, phase_diffs, N);

    // Calculate and print the mean of phase differences
    double mean_phase_diff = mean(phase_diffs, N);
    printf("Mean Phase Difference: %f radians\n", mean_phase_diff);

    return 0;
}
