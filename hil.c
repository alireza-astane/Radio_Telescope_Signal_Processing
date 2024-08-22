#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>

#define M_PI 3.14159265358979323846

// Define complex type
typedef double complex cplx;

// Helper function to print complex array (for debugging)
void print_array(double *array, int N) {
    for (int i = 0; i < N; i++) {
        printf("%lf,\n", array[i]);
    }
}



void print_array2(cplx *array, int N) {
    for (int i = 0; i < N; i++) {
        printf("(%lf, %lf)\n", creal(array[i]), cimag(array[i]));
    }
}


void print_array3(cplx *array, int N) {
    for (int i = 0; i < N; i++) {
        printf("%lf,\n", creal(array[i]));
    }
}

double mean(double *array,int size){
    double sum=0;
    for (size_t i = 0; i < size; i++)
    {
        sum += array[i];
    }
    return sum/size;
    
}


// Recursive FFT function
void fft(cplx *buf, cplx *out, int N) {
    if (N <= 1)
        for (int i = 0; i < N; i++)
        {
            out[i] = buf[i];
        }

    else{
        cplx buf_even[N/2];
        cplx buf_odd[N/2];
        cplx even[N/2];
        cplx odd[N/2];

        for (int i = 0; i < N/2; i++)
        {
            buf_even[i] = buf[2*i];
            buf_odd[i] = buf[2*i + 1];
        }
        fft(buf_even,even,N/2);
        fft(buf_odd,odd,N/2);


        for (int i = 0; i < N/2; i++) {
            cplx t = cexp(-2*I * M_PI * i / N) * odd[i];
            out[i] = even[i] + t;
            out[i + N/2] = even[i] - t;
        }
    }
        
}

// Recursive IFFT function
void ifft(cplx *buf, cplx *out, int N) {
    if (N <= 1)
        for (int i = 0; i < N; i++)
        {
            out[i] = buf[i];
        }

    else{
        cplx buf_even[N/2];
        cplx buf_odd[N/2];
        cplx even[N/2];
        cplx odd[N/2];

        for (int i = 0; i < N/2; i++)
        {
            buf_even[i] = buf[2*i];
            buf_odd[i] = buf[2*i + 1];
        }
        ifft(buf_even,even,N/2);
        ifft(buf_odd,odd,N/2);


        for (int i = 0; i < N/2; i++) {
            cplx t = cexp(2*I * M_PI * i / N) * odd[i];
            out[i] = (even[i] + t)/2;
            out[i + N/2] = (even[i] - t)/2;
        }
    }
}



// Hilbert transform
void hilbert_transform(cplx *signal, cplx *result, int N) {
    cplx *X = malloc(N * sizeof(cplx));
    cplx *H = malloc(N * sizeof(cplx));
    cplx *X_filtered = malloc(N * sizeof(cplx));
    cplx *analytic_signal = malloc(N * sizeof(cplx));

    for (int i = 0; i < N; i++) {
        X[i] = signal[i];
    }

    // fft(X, result, N, 1);
    fft(X, result, N);

    for (int i = 0; i < N; i++) {
        if (i == 0 || i == N / 2) {
            H[i] = 1;
        } else if (i < N / 2) {
            H[i] = 2;
        } else {
            H[i] = 0;
        }
    }

    for (int i = 0; i < N; i++) {
        X_filtered[i] = result[i] * H[i];
    }

    // ifft(X_filtered, analytic_signal, N, 1);
    ifft(X_filtered, analytic_signal, N);

    for (int i = 0; i < N; i++) {
        result[i] = analytic_signal[i];
    }

    free(X);
    free(H);
    free(X_filtered);
    free(analytic_signal);
}

// Phase unwrapping
void unwrap_phase(double *phase,double *unwrapped_phase, int N) {
    for (int i = 0; i < N; i++) {
        unwrapped_phase[i] = phase[i];
    }

    for (int i = 1; i < N; i++) {
        double delta = phase[i] - phase[i - 1];
        if (delta > M_PI) {
            for (int j = i; j < N; j++) {
                unwrapped_phase[j] -= 2 * M_PI;
            }
        } else if (delta < -M_PI) {
            for (int j = i; j < N; j++) {
                unwrapped_phase[j] += 2 * M_PI;
            }
        }
    }
}

// Main function for testing
int main() {
    double x_min = 0;
    double x_max = 10; // 2*pi for one full cycle of sine wave
    double rate = 1000;
    double frequency = 1000;
    double phi = 1;
    double amplitude = 1;


    int num_elements = 1024;

    cplx array1[num_elements];
    cplx array2[num_elements];

    // Generate random numbers and populate the array
    for (int i = 0; i < num_elements; ++i) {
        array1[i] = amplitude * sin((frequency*((double) i/rate) + x_min)); 
        array2[i] = amplitude * sin((frequency*((double) i/rate) + x_min) + phi); 
        // printf("%lf", amplitude * sin((frequency*((double) i/rate) + x_min) + phi));
        // printf("\n");
    }

    cplx hilbert1[num_elements];
    cplx hilbert2[num_elements];


    double phase1[num_elements];
    double phase2[num_elements];


    double unwrapped_phase1[num_elements];
    double unwrapped_phase2[num_elements];


    hilbert_transform(array1, hilbert1, num_elements);
    hilbert_transform(array2, hilbert2, num_elements);



    for (int i = 0; i < num_elements; ++i) {
        phase1[i] = carg(hilbert1[i]);
        phase2[i] = carg(hilbert2[i]);
    }


    unwrap_phase(phase1,unwrapped_phase1,num_elements);
    unwrap_phase(phase2,unwrapped_phase2,num_elements);

    double phase_diff[num_elements];


    for (int i = 0; i < num_elements; ++i) {
        phase_diff[i] = unwrapped_phase2[i] - unwrapped_phase1[i];
    }

    
    


 
    print_array3(array2, num_elements);
    printf("mean phase diff = %lf),\n", mean(phase_diff,num_elements));
    // print_array2(resuphase_difflt, num_elements);



    // printf("%lf,%lf",creal((1 + 2*I) * (3 - 0*I)),cimag((1 + 2*I) * (3 - 0*I)));



    // for (int i = 0; i < num_elements; i++) {
    //     printf("(%lf + %lfj),\n", creal(hilbert1[i]), cimag(hilbert1[i]));
    // }

 

    

    return 0;
}
