#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <fftw3.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define BUFFER_SIZE 1024
#define PI 3.14159265358979323846
#define PI2 6.283185307179586

void read_data(const char *filename, double *data, int *length) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("File opening failed");
        exit(EXIT_FAILURE);
    }

    int i = 0;
    while (fscanf(file, "%lf", &data[i]) != EOF && i < BUFFER_SIZE) {
        i++;
    }
    *length = i;

    fclose(file);
}

void hilbert_transform(double *data, double complex *hilbert, int length) {
    fftw_complex *in, *out;
    fftw_plan p_forward, p_inverse;

    in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * length);
    out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * length);

    for (int i = 0; i < length; i++) {
        in[i] = data[i] + I * 0.0;
    }

    p_forward = fftw_plan_dft_1d(length, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(p_forward);

    for (int i = 0; i < length; i++) {
        if (i == 0) {
            out[i] = out[i];
        } else if (i < (length + 1) / 2) {
            out[i] *= 2;
        } else {
            out[i] *= 0;
        }
    }

    p_inverse = fftw_plan_dft_1d(length, out, in, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(p_inverse);

    for (int i = 0; i < length; i++) {
        hilbert[i] = in[i] / length;
    }

    fftw_destroy_plan(p_forward);
    fftw_destroy_plan(p_inverse);
    fftw_free(in);
    fftw_free(out);
}

void print_complex_array(double complex *array, int length) {
    for (int i = 0; i < length; i++) {
        printf("%f + %fi\n", creal(array[i]), cimag(array[i]));
    }
}





void print_phase(double complex *array1,double complex *array2, int length) {
    for (int i = 0; i < length; i++) {
        // printf("%f + %fi\n", creal(array1[i]) - creal(array2[i]), cimag(array1[i]) - cimag(array2[i]));
        printf("%lf \n", atan2(cimag(array1[i]),creal(array1[i]) ) - atan2(cimag(array2[i]),creal(array2[i]) ));
    }
}




void cal_phase(double complex *array1,double complex *array2,double *phase_diff, int length) {
    for (int i = 0; i < length; i++) {


        phase_diff[i] = atan2(cimag(array1[i]),creal(array1[i]))- atan2(cimag(array2[i]),creal(array2[i]));
    }
}



void unwrap(double *phase, int length, double discontinuity_threshold) {
    // if (length < 2) {
    //     return;
    // }

    // double diff, correction = 0.0;
    // for (int i = 1; i < length; i++) {
    //     diff = phase[i] - phase[i - 1];
    //     if (fabs(diff) > discontinuity_threshold) {
    //         if (diff > 0) {
    //             correction -= 2 * PI;
    //         } else {
    //             correction += 2 * PI;
    //         }
    //     }
    //     phase[i] += correction;
    // }


    for (int i=1 ; i<length;i++){
        phase[i]  = fmod(phase[i],2*PI);
    }

}


void wrapToPi(double *angles, int length) {
    for (int i = 0; i < length; i++) {
        angles[i] = fmod(angles[i] + PI, PI2);
        if (angles[i] < 0) {
            angles[i] += PI2;
        }
        angles[i] -= PI;
    }
}


double mean(double *array, int length) {
    if (length == 0) {
        fprintf(stderr, "Array length is zero.\n");
        return 0.0;
    }

    double sum = 0.0;
    for (int i = 0; i < length; i++) {
        sum += array[i];
    }

    return sum / length;
}


int main() {
    double data1[BUFFER_SIZE];

    double data2[BUFFER_SIZE];



    double complex hilbert1[BUFFER_SIZE];
    double complex hilbert2[BUFFER_SIZE];
    int length;

    read_data("signal1.txt", data1, &length);
    hilbert_transform(data2, hilbert1, length);

    read_data("signal1.txt", data2, &length);
    hilbert_transform(data2, hilbert2, length);


    double phase_diff[length];

    cal_phase(hilbert1,hilbert2, phase_diff,length);
    unwrap(phase_diff,length,PI);
    

    FILE *myfile1;


      // Open a file for writing
    myfile1 = fopen("phase.txt", "w");
    if (myfile1 == NULL) {
        printf("Error creating file!\n");
        exit(1);
    }


    // Write the array elements to the file
    for (int i = 0; i < length; ++i) {
        fprintf(myfile1, "%lf\n", phase_diff[i]);
    }

    fclose(myfile1);



    printf("%lf",mean(phase_diff,length));

    return 0;
}