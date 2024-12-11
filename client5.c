#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <pthread.h>
#include <complex.h>
#include <math.h>
#include <fftw3.h>
#include <immintrin.h> 
#define BUFFER_SIZE 1024

#define SERVER_PORT1 12345
#define SERVER_PORT2 12346
#define PI 3.14159265358979323846
#define M_PI 3.14159265358979323846
#define sample_rate 20
#define freq 10 

typedef double dtype;
typedef double complex cplx;

typedef struct {
    int socket;
    dtype buffer[BUFFER_SIZE];
    pthread_mutex_t buffer_mutex;
} connection_info;




#define VECTOR_SIZE 4 // AVX2 processes 4 doubles at a time

int argmax(cplx arr[], int size) {
    int max_index = 0; // Start with the first element as the maximum
    for (int i = 1; i < size; i++) {
        if (cabs(arr[i]) > cabs(arr[max_index])) {
            max_index = i; // Update max_index if a larger value is found
        }
    }
    return max_index;
}





void roll(dtype *arr, int n, int shift) {
    if (n == 0) return;
    
    // Normalize the shift value to ensure it's within the bounds of the array
    shift = shift % n;
    if (shift < 0) {
        shift += n;
    }
    
    // Allocate a temporary array to hold the result
    dtype *temp = (dtype *)malloc(n * sizeof(dtype));

    // Copy the shifted elements into the temporary array
    memcpy(temp, &arr[n - shift], shift * sizeof(dtype));  // Copy the last 'shift' elements to the beginning
    memcpy(&temp[shift], arr, (n - shift) * sizeof(dtype));  // Copy the first 'n-shift' elements to the end

    // Copy the result back to the original array
    memcpy(arr, temp, n * sizeof(dtype));

    // Free the temporary array
    free(temp);
}






    

// Function to print an array
void print_array(dtype* array, int size) {
    printf("array ([");
    for (int i = 0; i < size-1; i++) {
        printf("%f ,", array[i]);
    }
    printf("%f ])",array[size-1]);
    printf("\n");
}



// Function to print a complex array
void print_carray(cplx* array, int size) {
    printf("array ([");
    for (int i = 0; i < size-1; i++) {
        printf("%f ,", creal(array[i]));
    }
    printf("%f ])",creal(array[size-1]));
    printf("\n");
}



void hilbert_transform(dtype *signal1,dtype *signal2,dtype *dot) {
    // print_array(signal1,N);
    // print_array(signal2,N);

    cplx *transformed1 = malloc(BUFFER_SIZE * sizeof(cplx));
    cplx *transformed2 = malloc(BUFFER_SIZE * sizeof(cplx));

    fftw_plan plan1;
    fftw_plan plan2;

    cplx *X1 = malloc(BUFFER_SIZE * sizeof(cplx));
    cplx *X2 = malloc(BUFFER_SIZE * sizeof(cplx));



    dtype *means_phase_diff = malloc(sizeof(dtype));

    int *shift = malloc(sizeof(int));





    for (int i = 0; i < BUFFER_SIZE; i++) {
        X1[i] = signal1[i];
        X2[i] = signal2[i];
    }



    // Create a plan for the FFT
    plan1 = fftw_plan_dft_1d(BUFFER_SIZE, X1, transformed1, FFTW_FORWARD, FFTW_ESTIMATE);
    plan2 = fftw_plan_dft_1d(BUFFER_SIZE, X2, transformed2, FFTW_FORWARD, FFTW_ESTIMATE);

    // plan1 = ftwf_plan_dft_1d(BUFFER_SIZE, X1, transformed1, FFTW_FORWARD, FFTW_ESTIMATE);
    // plan2 = ftwf_plan_dft_1d(BUFFER_SIZE, X2, transformed2, FFTW_FORWARD, FFTW_ESTIMATE);




    // Execute the FFT
    fftw_execute(plan1);
    fftw_execute(plan2);




    

    *means_phase_diff = carg(transformed1[argmax(transformed1,BUFFER_SIZE)]) - carg(transformed2[argmax(transformed2,BUFFER_SIZE)]);


    *shift = (int)round(*means_phase_diff*sample_rate/freq);


    printf("phase diff = %lf\n",*means_phase_diff);


    printf("shift = %d\n",*shift);

    roll(signal1,BUFFER_SIZE,*shift);

    *dot = 0;

    for (int i = 0; i < BUFFER_SIZE; i++)
    {
        *dot +=signal1[i] * signal2[i];
    }

    *dot /= BUFFER_SIZE;

    fftw_destroy_plan(plan1);
    fftw_destroy_plan(plan2);
    free(transformed1);
    free(transformed2);
    free(means_phase_diff);
    free(shift);
    free(X1);
    free(X2);



}



void *calculate_phase_difference(void *arg) {
    connection_info *conn_info1 = ((connection_info **)arg)[0];
    connection_info *conn_info2 = ((connection_info **)arg)[1];
    FILE *file = fopen("product.txt", "w");
    if (!file) {
        perror("File opening failed");
        return NULL;
    }

    dtype *dot = malloc(sizeof(dtype));;
    dtype sampling_rate = 1024.0;  // Adjust according to actual sampling rate

    // int i = 0; 
    while (1) {
        // i += 1; 
        pthread_mutex_lock(&conn_info1->buffer_mutex);
        pthread_mutex_lock(&conn_info2->buffer_mutex);


        hilbert_transform(conn_info1->buffer,conn_info2->buffer,dot);
        // printf("%f \n",*phase_diff);
        
        fprintf(file, "%lf\n", *dot);
        

        pthread_mutex_unlock(&conn_info2->buffer_mutex);
        pthread_mutex_unlock(&conn_info1->buffer_mutex);
        // usleep(1000000 / sampling_rate);  // TODO ?????
    }

    fclose(file);
    return NULL;
}


void *receive_data(void *arg) {
    connection_info *conn_info = (connection_info *)arg;
    int bytes_received;

    while ((bytes_received = recv(conn_info->socket, conn_info->buffer, sizeof(conn_info->buffer), 0)) > 0) {
        pthread_mutex_lock(&conn_info->buffer_mutex);
        // Process the received data
        pthread_mutex_unlock(&conn_info->buffer_mutex);
    }

    if (bytes_received < 0) {
        perror("Receive failed");
    }

    close(conn_info->socket);
    return NULL;
}

int main() {




    int sock1, sock2;
    struct sockaddr_in server_addr1, server_addr2;

    // Create sockets
    if ((sock1 = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        perror("Socket creation failed");
        exit(EXIT_FAILURE);
    }
    if ((sock2 = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        perror("Socket creation failed");
        close(sock1);
        exit(EXIT_FAILURE);
    }

    // Setup server addresses
    server_addr1.sin_family = AF_INET;
    server_addr1.sin_port = htons(SERVER_PORT1);
    if (inet_pton(AF_INET, "127.0.0.1", &server_addr1.sin_addr) <= 0) {
        perror("Invalid address/ Address not supported");
        close(sock1);
        close(sock2);
        exit(EXIT_FAILURE);
    }

    server_addr2.sin_family = AF_INET;
    server_addr2.sin_port = htons(SERVER_PORT2);
    if (inet_pton(AF_INET, "127.0.0.1", &server_addr2.sin_addr) <= 0) {
        perror("Invalid address/ Address not supported");
        close(sock1);
        close(sock2);
        exit(EXIT_FAILURE);
    }

    // Connect to servers
    if (connect(sock1, (struct sockaddr *)&server_addr1, sizeof(server_addr1)) < 0) {
        perror("Connection failed");
        close(sock1);
        close(sock2);
        exit(EXIT_FAILURE);
    }

    if (connect(sock2, (struct sockaddr *)&server_addr2, sizeof(server_addr2)) < 0) {
        perror("Connection failed");
        close(sock1);
        close(sock2);
        exit(EXIT_FAILURE);
    }

    printf("Connected to server 1 on port %d\n", SERVER_PORT1);
    printf("Connected to server 2 on port %d\n", SERVER_PORT2);

    // Create connection info structures
    connection_info conn_info1 = {sock1, {0}, PTHREAD_MUTEX_INITIALIZER};
    connection_info conn_info2 = {sock2, {0}, PTHREAD_MUTEX_INITIALIZER};

    connection_info *conn_infos[2] = {&conn_info1, &conn_info2};

    // Create threads to receive data
    pthread_t thread_id1, thread_id2, process_thread1, process_thread2, phase_diff_thread;
    pthread_create(&thread_id1, NULL, receive_data, &conn_info1);
    pthread_create(&thread_id2, NULL, receive_data, &conn_info2);
    pthread_create(&phase_diff_thread, NULL, calculate_phase_difference, conn_infos);

    // Wait for the threads to finish
    pthread_join(thread_id1, NULL);
    pthread_join(thread_id2, NULL);
    pthread_join(phase_diff_thread, NULL);

    return 0;
}