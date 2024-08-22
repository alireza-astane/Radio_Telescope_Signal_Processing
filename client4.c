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

#define SERVER_PORT1 12345
#define SERVER_PORT2 12346
#define BUFFER_SIZE 1024
#define PI 3.14159265358979323846
#define M_PI 3.14159265358979323846

// Define complex type
typedef double complex cplx;

typedef struct {
    int socket;
    double buffer[BUFFER_SIZE];
    pthread_mutex_t buffer_mutex;
} connection_info;

#define VECTOR_SIZE 4 // AVX2 processes 4 doubles at a time

    cplx *H;
    
void hilbert_transform(double *signal1,double *signal2,double means_phase_diff, int N) {

    cplx transformed1[N];
    cplx transformed2[N];

    cplx *X1 = malloc(N * sizeof(cplx));
    cplx *X2 = malloc(N * sizeof(cplx));

    cplx *analytic_signal1 = malloc(N * sizeof(cplx));
    cplx *analytic_signal2 = malloc(N * sizeof(cplx));


    for (int i = 0; i < N; i++) {
        X1[i] = signal1[i];
        X2[i] = signal2[i];
    }

    fftw_plan plan1;
    fftw_plan plan2;

    // Create a plan for the FFT
    plan1 = fftw_plan_dft_1d(N, X1, transformed1, FFTW_FORWARD, FFTW_ESTIMATE);
    plan2 = fftw_plan_dft_1d(N, X2, transformed2, FFTW_FORWARD, FFTW_ESTIMATE);

    // Execute the FFT
    fftw_execute(plan1);
    fftw_execute(plan2);

    // // Destroy the FFT plan
    // fftw_destroy_plan(plan);   ????


    for (int i = 0; i < N; i++) {
        X1[i] = transformed1[i] * H[i];   // ??? replace with x 
        X2[i] = transformed2[i] * H[i];
    }


    plan1 = fftw_plan_dft_1d(N, X1, analytic_signal1, FFTW_BACKWARD, FFTW_ESTIMATE);
    plan2 = fftw_plan_dft_1d(N, X2, analytic_signal2, FFTW_BACKWARD, FFTW_ESTIMATE);

    // Execute the IFFT
    fftw_execute(plan1);
    fftw_execute(plan2);


    // Normalize the result
    for (int i = 0; i < N; ++i) {
        analytic_signal1[i] /= N; // Normalize the result
        analytic_signal2[i] /= N; // Normalize the result
    }
    // Destroy the FFT plan
    fftw_destroy_plan(plan1);
    fftw_destroy_plan(plan2);

    // compute_ifft(X_filtered, analytic_signal, N);

    free(X1);
    free(X2);

    double phase1[N];
    double phase2[N];

    
    for (int i = 0; i < N; ++i) {
        phase1[i] = carg(analytic_signal1[i]);
        phase2[i] = carg(analytic_signal2[i]);

    }
    double unwrapped_phase1[N];
    double unwrapped_phase2[N];


    // unwrap 
    __m256d pi = _mm256_set1_pd(M_PI);
    __m256d neg_pi = _mm256_set1_pd(-M_PI);

    // Initialize the first value
    unwrapped_phase1[0] = phase1[0];
    unwrapped_phase2[0] = phase2[0];


    // Process in chunks of VECTOR_SIZE
    for (int i = 1; i < N; i += VECTOR_SIZE) {
        __m256d phase_vec1 = _mm256_loadu_pd(&phase1[i]);
        __m256d phase_vec2 = _mm256_loadu_pd(&phase2[i]);

        __m256d prev_unwrapped_vec1 = _mm256_set1_pd(unwrapped_phase1[i - 1]);
        __m256d prev_unwrapped_vec2 = _mm256_set1_pd(unwrapped_phase2[i - 1]);


        // Compute phase differences
        __m256d delta1 = _mm256_sub_pd(phase_vec1, prev_unwrapped_vec1);
        __m256d delta2 = _mm256_sub_pd(phase_vec2, prev_unwrapped_vec2);

        // Unwrap phases: ensure delta is within [-π, π]
        __m256d mask1 = _mm256_cmp_pd(delta1, pi, _CMP_GT_OS);
        __m256d mask2 = _mm256_cmp_pd(delta2, pi, _CMP_GT_OS);

        __m256d adjustment1 = _mm256_blendv_pd(neg_pi, pi, mask1);
        __m256d adjustment2 = _mm256_blendv_pd(neg_pi, pi, mask2);

        delta1 = _mm256_sub_pd(delta1, adjustment1);
        delta2 = _mm256_sub_pd(delta2, adjustment2);


        mask1 = _mm256_cmp_pd(delta1, neg_pi, _CMP_LT_OS);
        mask2 = _mm256_cmp_pd(delta2, neg_pi, _CMP_LT_OS);

        adjustment1 = _mm256_blendv_pd(pi, neg_pi, mask1);
        adjustment2 = _mm256_blendv_pd(pi, neg_pi, mask2);

        delta1 = _mm256_add_pd(delta1, adjustment1);
        delta2 = _mm256_add_pd(delta2, adjustment2);

        // Compute the unwrapped values
        __m256d unwrapped_vec1 = _mm256_add_pd(prev_unwrapped_vec1, delta1);
        __m256d unwrapped_vec2 = _mm256_add_pd(prev_unwrapped_vec2, delta2);


        // Store the result
        _mm256_storeu_pd(&unwrapped_phase1[i], unwrapped_vec1);
        _mm256_storeu_pd(&unwrapped_phase2[i], unwrapped_vec2);

    }




    // unwrap_phases_avx(phase1,unwrapped_phase1,N);
    // unwrap_phases_avx(phase2,unwrapped_phase2,N);


    // diff and mean 

    __m256d sum_vec = _mm256_setzero_pd();
    int i;


    // Process elements in chunks of VECTOR_SIZE
    for (i = 1; i < N ; i += VECTOR_SIZE) {
        __m256d data_vec1 = _mm256_loadu_pd(&unwrapped_phase1[i]);
        __m256d data_vec2 = _mm256_loadu_pd(&unwrapped_phase2[i]);

        __m256d diff_avx = _mm256_sub_pd(data_vec1,data_vec2);

        sum_vec = _mm256_add_pd(sum_vec, diff_avx);
    }

    // Horizontal sum to get the total sum
    double sum[4];
    _mm256_storeu_pd(sum, sum_vec);

    double total_sum = sum[0] + sum[1] + sum[2] + sum[3];

    means_phase_diff =  total_sum / N;

}



void *calculate_phase_difference(void *arg) {
    connection_info *conn_info1 = ((connection_info **)arg)[0];
    connection_info *conn_info2 = ((connection_info **)arg)[1];
    FILE *file = fopen("phase_difference.txt", "w");
    if (!file) {
        perror("File opening failed");
        return NULL;
    }

    double phase_diff;
    double sampling_rate = 1000.0;  // Adjust according to actual sampling rate

    while (1) {
        pthread_mutex_lock(&conn_info1->buffer_mutex);
        pthread_mutex_lock(&conn_info2->buffer_mutex);

        double unwrapped_phase1[BUFFER_SIZE];
        double unwrapped_phase2[BUFFER_SIZE];

        double diff[BUFFER_SIZE];

        hilbert_transform(conn_info1->buffer,conn_info2->buffer,phase_diff,BUFFER_SIZE);
        
        fprintf(file, "%lf\n", phase_diff);
        

        pthread_mutex_unlock(&conn_info2->buffer_mutex);
        pthread_mutex_unlock(&conn_info1->buffer_mutex);
        // usleep(1000000 / sampling_rate);  TODO ?????
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
    cplx *H = malloc(BUFFER_SIZE * sizeof(cplx));
    for (int i = 0; i < BUFFER_SIZE; i++) {
        if (i == 0 || i == BUFFER_SIZE / 2) {
            H[i] = 1;
        } else if (i < BUFFER_SIZE / 2) {
            H[i] = 2;
        } else {
            H[i] = 0;
        }
    }



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
    // pthread_create(&process_thread1, NULL, process_signal, &conn_info1);
    // pthread_create(&process_thread2, NULL, process_signal, &conn_info2);
    pthread_create(&phase_diff_thread, NULL, calculate_phase_difference, conn_infos);

    // Wait for the threads to finish
    pthread_join(thread_id1, NULL);
    pthread_join(thread_id2, NULL);
    // pthread_join(process_thread1, NULL);
    // pthread_join(process_thread2, NULL);
    pthread_join(phase_diff_thread, NULL);

    return 0;
}