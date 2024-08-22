#include <stdio.h>
#include <stdlib.h>
#include <math.h>>

#define PI 3.14159265358979323846


// Function to compute the Hilbert transform
void hilbertTransform(double* input, double* output, int length) {
    for (int n = 0; n < length; ++n) {
        double sum = 0.0;
        for (int k = 0; k < length; ++k) {
            if (k != n){
            sum += input[k] / (PI * (n - k));}
        }
        output[n] = sum;
    }
}

int main() {
    FILE *signalFile1;
    FILE *signalFile2;
    FILE *hilberFile1;
    FILE *hilberFile2;


    int length = 10000;
    double array1[length];
    double array2[length];

    double hilbertArray1[length];
    double hilbertArray2[length];


    signalFile1 = fopen("signal1.txt", "r");
    signalFile2 = fopen("signal2.txt", "r");


    if (signalFile1 == NULL) {
        printf("Error opening file. Exiting.\n");
        return 1;
    }


        if (signalFile2 == NULL) {
        printf("Error opening file. Exiting.\n");
        return 1;
    }

    for (int i = 0; i < length; i++) {
        fscanf(signalFile1, "%lf", &array1[i]);
        fscanf(signalFile2, "%lf", &array2[i]);
    }

    fclose(signalFile1);
    fclose(signalFile2);

    hilbertTransform(array1,hilbertArray1,length);
    hilbertTransform(array2,hilbertArray2,length); 




    // for (int i = 0; i < 10000; i++) {
    //     printf("Number is: %lf\n", hilbertArray[i]);
    // }


    hilberFile1 = fopen("hilbert1.txt", "w");
    hilberFile2 = fopen("hilbert2.txt", "w");

    if (hilberFile1 == NULL) {
        printf("Error opening file. Exiting.\n");
        return 1;
    }

    if (hilberFile2 == NULL) {
        printf("Error opening file. Exiting.\n");
        return 1;
    }   

    for (int i = 0; i < length; ++i) {
        fprintf(hilberFile1, "%lf\n", hilbertArray1[i]);
        fprintf(hilberFile2, "%lf\n", hilbertArray2[i]);
    }


    fclose(hilberFile1);
    fclose(hilberFile2);

    printf("Random array saved to 'random_array.txt'.\n");
    


    return 0;
}
