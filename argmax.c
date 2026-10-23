#include <stdio.h>

int argmax(double arr[], int size) {
    int max_index = 0; // Start with the first element as the maximum
    for (int i = 1; i < size; i++) {
        if (arr[i] > arr[max_index]) {
            max_index = i; // Update max_index if a larger value is found
        }
    }
    return max_index;
}

int main() {
    double arr[] = {1.1, 3.2, 7.7, 0.1, 5.2};
    int size = sizeof(arr) / sizeof(arr[0]);

    int index_of_max = argmax(arr, size);
    printf("The index of the maximum element is: %d\n", index_of_max);

    return 0;
}