#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define PI 3.14159265358979323846

int main(void) {

    // Define the range of x-values to plot
    double x_min = 0;
    double x_max = 10; // 2*pi for one full cycle of sine wave
    double rate = 1000;
    double frequency = 1;
    double phi = 1;
    double amplitude = 1;


    int num_elements = rate * (x_max- x_min); // Change this to the desired array size
    double array1[num_elements];
    double array2[num_elements];

    FILE *myfile1;
    FILE *myfile2;


    // Generate random numbers and populate the array
    for (int i = 0; i < num_elements; ++i) {
        array1[i] = amplitude * sin((frequency*((double) i/rate) + x_min)); 
        array2[i] = amplitude * sin((frequency*((double) i/rate) + x_min) + phi); 
        // printf("%lf", amplitude * sin((frequency*((double) i/rate) + x_min) + phi));
        // printf("\n");
    }

    // Open a file for writing
    myfile1 = fopen("signal1.txt", "w");
    if (myfile1 == NULL) {
        printf("Error creating file!\n");
        exit(1);
    }



    myfile2 = fopen("signal2.txt", "w");
    if (myfile2 == NULL) {
        printf("Error creating file!\n");
        exit(1);
    }



    // Write the array elements to the file
    for (int i = 0; i < num_elements; ++i) {
        fprintf(myfile1, "%lf\n", array1[i]);
        fprintf(myfile2, "%lf\n", array2[i]);
    }



    fclose(myfile1);
    fclose(myfile2);
    printf("Random array saved to 'random_array.txt'.\n");

    return 0;
}






