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



// Function to unwrap phase angles using AVX
void unwrap_phases_avx(double* phases, double* unwrapped, int n) {
    __m256d pi = _mm256_set1_pd(M_PI);
    __m256d neg_pi = _mm256_set1_pd(-M_PI);

    // Initialize the first value
    unwrapped[0] = phases[0];

    // Process in chunks of VECTOR_SIZE
    for (int i = 1; i < n; i += VECTOR_SIZE) {
        __m256d phase_vec = _mm256_loadu_pd(&phases[i]);
        __m256d prev_unwrapped_vec = _mm256_set1_pd(unwrapped[i - 1]);

        // Compute phase differences
        __m256d delta = _mm256_sub_pd(phase_vec, prev_unwrapped_vec);

        // Unwrap phases: ensure delta is within [-π, π]
        __m256d mask = _mm256_cmp_pd(delta, pi, _CMP_GT_OS);
        __m256d adjustment = _mm256_blendv_pd(neg_pi, pi, mask);
        delta = _mm256_sub_pd(delta, adjustment);

        mask = _mm256_cmp_pd(delta, neg_pi, _CMP_LT_OS);
        adjustment = _mm256_blendv_pd(pi, neg_pi, mask);
        delta = _mm256_add_pd(delta, adjustment);

        // Compute the unwrapped values
        __m256d unwrapped_vec = _mm256_add_pd(prev_unwrapped_vec, delta);

        // Store the result
        _mm256_storeu_pd(&unwrapped[i], unwrapped_vec);
    }
}

// Function to compute the mean of an array using AVX
double mean_avx(double* array, int n) {
    __m256d sum_vec = _mm256_setzero_pd();
    int i;

    // Process elements in chunks of VECTOR_SIZE
    for (i = 0; i <= n - VECTOR_SIZE; i += VECTOR_SIZE) {
        __m256d data_vec = _mm256_loadu_pd(&array[i]);
        sum_vec = _mm256_add_pd(sum_vec, data_vec);
    }

    // Horizontal sum to get the total sum
    double sum[4];
    _mm256_storeu_pd(sum, sum_vec);

    double total_sum = sum[0] + sum[1] + sum[2] + sum[3];

    // Handle any remaining elements
    for (; i < n; ++i) {
        total_sum += array[i];
    }

    return total_sum / n;
}


// double mean(double* array, int n) {
//     double sum = 0.0;
//     for (int i = 0; i < n; ++i) {
//         sum += array[i];
//     }
//     return sum / n;
// }

// Function to perform FFT using FFTW3 with double complex arrays  commented !!!!!
void compute_fft(double complex* input, double complex* output, int n) {
    fftw_plan plan;
    // fftw_complex* fftw_input = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n);
    // fftw_complex* fftw_output = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n);

    // Convert double complex to fftw_complex
    // for (int i = 0; i < n; ++i) {
    //     fftw_input[i] = creal(input[i]) + I * cimag(input[i]);
    // }

    // Create a plan for the FFT
    plan = fftw_plan_dft_1d(n, input, output, FFTW_FORWARD, FFTW_ESTIMATE);

    // Execute the FFT
    fftw_execute(plan);

    // Convert fftw_complex to double complex
    // for (int i = 0; i < n; ++i) {
    //     output[i] = fftw_output[i];
    // }

    // Destroy the FFT plan
    fftw_destroy_plan(plan);

    // Free the temporary FFTW arrays
    // fftw_free(fftw_input);
    // fftw_free(fftw_output);
}

// Function to perform IFFT using FFTW3 with double complex arrays
void compute_ifft(double complex* input, double complex* output, int n) {
    fftw_plan plan;
    // cplx* fftw_input = (cplx*) fftw_malloc(sizeof(cplx) * n);
    // cplx* fftw_output = (cplx*) fftw_malloc(sizeof(cplx) * n);

    // // Convert double complex to fftw_complex
    // for (int i = 0; i < n; ++i) {
    //     fftw_input[i] = creal(input[i]) + I * cimag(input[i]);
    // }

    // Create a plan for the IFFT
    // plan = fftw_plan_dft_1d(n, fftw_input, fftw_output, FFTW_BACKWARD, FFTW_ESTIMATE);
    plan = fftw_plan_dft_1d(n, input, output, FFTW_BACKWARD, FFTW_ESTIMATE);


    // Execute the IFFT
    fftw_execute(plan);

    // Convert fftw_complex to double complex and normalize the result
    for (int i = 0; i < n; ++i) {
        // output[i] = creal(fftw_output[i])+ I * cimag(fftw_output[i]); commented!!!
        output[i] /= n; // Normalize the result
    }

    // Destroy the FFT plan
    fftw_destroy_plan(plan);

    // Free the temporary FFTW arrays
    // fftw_free(fftw_input); commented!!!!
    // fftw_free(fftw_output);
}

// // Recursive FFT function
// void fft(cplx *buf, cplx *out, int N) {
//     if (N <= 1)
//         for (int i = 0; i < N; i++)
//         {
//             out[i] = buf[i];
//         }

//     else{
//         cplx buf_even[N/2];
//         cplx buf_odd[N/2];
//         cplx even[N/2];
//         cplx odd[N/2];

//         for (int i = 0; i < N/2; i++)
//         {
//             buf_even[i] = buf[2*i];
//             buf_odd[i] = buf[2*i + 1];
//         }
//         fft(buf_even,even,N/2);
//         fft(buf_odd,odd,N/2);


//         for (int i = 0; i < N/2; i++) {
//             cplx t = cexp(-2*I * M_PI * i / N) * odd[i];
//             out[i] = even[i] + t;
//             out[i + N/2] = even[i] - t;
//         }
//     }
        
// }

// // Recursive IFFT function
// void ifft(cplx *buf, cplx *out, int N) {
//     if (N <= 1)
//         for (int i = 0; i < N; i++)
//         {
//             out[i] = buf[i];
//         }

//     else{
//         cplx buf_even[N/2];
//         cplx buf_odd[N/2];
//         cplx even[N/2];
//         cplx odd[N/2];

//         for (int i = 0; i < N/2; i++)
//         {
//             buf_even[i] = buf[2*i];
//             buf_odd[i] = buf[2*i + 1];
//         }
//         ifft(buf_even,even,N/2);
//         ifft(buf_odd,odd,N/2);


//         for (int i = 0; i < N/2; i++) {
//             cplx t = cexp(2*I * M_PI * i / N) * odd[i];
//             out[i] = (even[i] + t)/2;
//             out[i + N/2] = (even[i] - t)/2;
//         }
//     }
// }



// // Hilbert transform
// void hilbert_transform(cplx *signal, cplx *result, int N) {
//     cplx *X = malloc(N * sizeof(cplx));
//     cplx *H = malloc(N * sizeof(cplx));
//     cplx *X_filtered = malloc(N * sizeof(cplx));
//     cplx *analytic_signal = malloc(N * sizeof(cplx));

//     for (int i = 0; i < N; i++) {
//         X[i] = signal[i];
//     }

//     // fft(X, result, N, 1);
//     fft(X, result, N);

//     for (int i = 0; i < N; i++) {
//         if (i == 0 || i == N / 2) {
//             H[i] = 1;
//         } else if (i < N / 2) {
//             H[i] = 2;
//         } else {
//             H[i] = 0;
//         }
//     }

//     for (int i = 0; i < N; i++) {
//         X_filtered[i] = result[i] * H[i];
//     }

//     // ifft(X_filtered, analytic_signal, N, 1);
//     ifft(X_filtered, analytic_signal, N);

//     for (int i = 0; i < N; i++) {
//         result[i] = analytic_signal[i];
//     }

//     free(X);
//     free(H);
//     free(X_filtered);
//     free(analytic_signal);
// }

// // Phase unwrapping
// void unwrap_phase(double *phase,double *unwrapped_phase, int N) {
//     for (int i = 0; i < N; i++) {
//         unwrapped_phase[i] = phase[i];
//     }

//     for (int i = 1; i < N; i++) {
//         double delta = phase[i] - phase[i - 1];
//         if (delta > M_PI) {
//             for (int j = i; j < N; j++) {
//                 unwrapped_phase[j] -= 2 * M_PI;
//             }
//         } else if (delta < -M_PI) {
//             for (int j = i; j < N; j++) {
//                 unwrapped_phase[j] += 2 * M_PI;
//             }
//         }
//     }
// }
    cplx *H;
    
void hilbert_transform(double *signal,double *unwrapped_phase, int N) {

    double complex transformed[N];
    cplx *X = malloc(N * sizeof(cplx));

    cplx *X_filtered = malloc(N * sizeof(cplx));
    cplx *analytic_signal = malloc(N * sizeof(cplx));

    for (int i = 0; i < N; i++) {
        X[i] = signal[i];
    }

    // fft(X, result, N, 1);
    // fft(X, transformed, N);
    compute_fft(X, transformed, N);





    for (int i = 0; i < N; i++) {
        X_filtered[i] = transformed[i] * H[i];
    }

    // ifft(X_filtered, analytic_signal, N, 1);
    // ifft(X_filtered, analytic_signal, N);
    compute_ifft(X_filtered, analytic_signal, N);

    // for (int i = 0; i < N; i++) {
    //     transformed[i] = analytic_signal[i];
    // }

    free(X);
    // free(H);
    free(X_filtered);
    // free(analytic_signal);

    double phase[N];
    
    for (int i = 0; i < N; ++i) {
        phase[i] = carg(analytic_signal[i]);
    }

    unwrap_phases_avx(phase,unwrapped_phase,N);
    
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

// void *process_signal(void *arg) {
//     connection_info *conn_info = (connection_info *)arg;
//     double complex transformed[BUFFER_SIZE];
//     double phase[BUFFER_SIZE];
//     double sampling_rate = 1000000.0;  // Adjust according to actual sampling rate

//     while (1) {
//         pthread_mutex_lock(&conn_info->buffer_mutex);
//         for (int i = 0; i < BUFFER_SIZE; i++) {
//             transformed[i] = hilbert_transform(conn_info->buffer, BUFFER_SIZE)[i];
//             phase[i] = carg(transformed[i]);  // Get phase
//         }
//         pthread_mutex_unlock(&conn_info->buffer_mutex);
//         usleep(1000000 / sampling_rate);
//     }

//     return NULL;
// }

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


        
        hilbert_transform(conn_info1->buffer,unwrapped_phase1, BUFFER_SIZE);
        hilbert_transform(conn_info2->buffer,unwrapped_phase2, BUFFER_SIZE);

        for (size_t i = 0; i < BUFFER_SIZE; i++)
        {
            diff[i] = unwrapped_phase2[i] - unwrapped_phase1[i];
        }


        
        // for (int i = 0; i < BUFFER_SIZE; i++) {
        //     phase_diff = mean(diff,BUFFER_SIZE);
        //     fprintf(file, "%lf\n", phase_diff);
        // }

           
        phase_diff = mean_avx(diff,BUFFER_SIZE);
        











        fprintf(file, "%lf\n", phase_diff);
        

        pthread_mutex_unlock(&conn_info2->buffer_mutex);
        pthread_mutex_unlock(&conn_info1->buffer_mutex);
        // usleep(1000000 / sampling_rate);  TODO ?????
    }

    fclose(file);
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