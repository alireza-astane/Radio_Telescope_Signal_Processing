#include <stdio.h>
#include <math.h>

#define PI 3.14159265358979323846

int main() {
  // Define the range of x-values to plot
  double x_min = 0;
  double x_max = 2 * PI; // 2*pi for one full cycle of sine wave
  double step = 0.1;

  // Calculate y-values for each x-value
  for (double x = x_min; x <= x_max; x += step) {
    double y = sin(x);
    printf("(%f, %f)\n", x, y);
  }

  return 0;
}/home/alireza-astane/Desktop/astro/sin.c