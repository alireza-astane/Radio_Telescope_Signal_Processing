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
#define sample_rate 20
#define freq 10 

// Define complex type
typedef double complex cplx;

typedef struct {
    int socket;
    double buffer[BUFFER_SIZE];
    pthread_mutex_t buffer_mutex;
} connection_info;

#define VECTOR_SIZE 4 // AVX2 processes 4 doubles at a time

    cplx *H;



#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>



#include <immintrin.h>
#include <math.h>

#define VECTOR_SIZE 4 // AVX2 processes 4 doubles at a time

typedef __m128i v4i;

v4i prefix(v4i x) {
    // x = 1, 2, 3, 4
    x = _mm_add_epi32(x, _mm_slli_si128(x, 4));
    // x = 1, 2, 3, 4
    //   + 0, 1, 2, 3
    //   = 1, 3, 5, 7
    x = _mm_add_epi32(x, _mm_slli_si128(x, 8));
    // x = 1, 3, 5, 7
    //   + 0, 0, 1, 3
    //   = 1, 3, 6, 10
    return x;
}

// __m256d cumsum(__m256d x) {
//     x = _mm256_add_pd(x, _mm256_slli_pd(x, 4));
//     x = _mm256_add_pd(x, _mm256_slli_si256(x, 8));
//     return x;
// }


void roll(double *arr, int n, int shift) {
    if (n == 0) return;
    
    // Normalize the shift value to ensure it's within the bounds of the array
    shift = shift % n;
    if (shift < 0) {
        shift += n;
    }
    
    // Allocate a temporary array to hold the result
    double *temp = (double *)malloc(n * sizeof(double));

    // Copy the shifted elements into the temporary array
    memcpy(temp, &arr[n - shift], shift * sizeof(double));  // Copy the last 'shift' elements to the beginning
    memcpy(&temp[shift], arr, (n - shift) * sizeof(double));  // Copy the first 'n-shift' elements to the end

    // Copy the result back to the original array
    memcpy(arr, temp, n * sizeof(double));

    // Free the temporary array
    free(temp);
}



// Phase unwrapping
void unwrap_phase3(double *phase,double *unwrapped_phase, int N) {
    __m256d pi = _mm256_set1_pd(M_PI);
    __m256d neg_pi = _mm256_set1_pd(-M_PI);
    __m256d two_pi = _mm256_set1_pd(2*M_PI);
    __m256d one = _mm256_set1_pd(1);
    __m256d zero = _mm256_set1_pd(0);
    __m256d minus_one = _mm256_set1_pd(-1);

    // Initialize the first value
    unwrapped_phase[0] = phase[0];
    __m256d sum =  _mm256_set1_pd(0);

    // Process in chunks of VECTOR_SIZE
    for (int i = 1; i < N; i += VECTOR_SIZE) {
        __m256d d = _mm256_set1_pd(0);

        __m256d phase_vec = _mm256_loadu_pd(&phase[i]);
        __m256d prev_unwrapped_vec = _mm256_loadu_pd(&phase[i-1]);

        // Compute phase differences
        __m256d delta = _mm256_sub_pd(phase_vec, prev_unwrapped_vec);
        __m256d mask1 = _mm256_cmp_pd(delta, pi, _CMP_GT_OS);
        __m256d mask2 = _mm256_cmp_pd(delta, neg_pi, _CMP_LT_OS);


        __m256d adjustment = _mm256_blendv_pd(_mm256_blendv_pd(zero,one,mask2),minus_one, mask1);
        adjustment = _mm256_add_pd(  _mm256_castsi256_pd(_mm256_castsi128_si256(prefix(_mm256_castsi256_si128(_mm256_castpd_si256(adjustment))))) ,sum);
        sum  =  _mm256_set1_pd(adjustment[3]);

        adjustment = _mm256_mul_pd(adjustment,two_pi);

        // Compute the unwrapped values
        __m256d unwrapped_vec = _mm256_add_pd(phase_vec, adjustment);

        // Store the result
        _mm256_storeu_pd(&unwrapped_phase[i], unwrapped_vec);


    
        

        
        
    }







    for (int i = 0; i < N; i++) {
        unwrapped_phase[i] = phase[i];
    }
    int ks[N];
    int k = 0;
    ks[0] = 0;

    for (int i = 1; i < N; i++)
    {
        double delta = phase[i] - phase[i - 1];
        int d = (delta >= M_PI) ? -1 : (delta <= -M_PI ? 1 : 0 ) ;
        k = k + d ;
        ks[i] = k;
    }
    
    for (int i = 0; i < N; i++)
    {
        unwrapped_phase[i] += ks[i] * 2* M_PI;
    }
}



// Phase unwrapping
void unwrap_phase2(double *phase,double *unwrapped_phase, int N) {
    for (int i = 0; i < N; i++) {
        unwrapped_phase[i] = phase[i];
    }
    int ks[N];
    int k = 0;
    ks[0] = 0;

    for (int i = 1; i < N; i++)
    {
        double delta = phase[i] - phase[i - 1];
        int d = (delta > M_PI) ? -1 : (delta < -M_PI ? 1 : 0 ) ;
        k = k + d ;
        ks[i] = k;
    }
    
    for (int i = 0; i < N; i++)
    {
        unwrapped_phase[i] += ks[i] * 2* M_PI;
    }
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



// Function to unwrap phase angles using AVX
void unwrap_phases_avx(double* phases, double* unwrapped, int n) {
    __m256d pi = _mm256_set1_pd(M_PI);
    __m256d neg_pi = _mm256_set1_pd(-M_PI);

    // Initialize the first value
    unwrapped[0] = phases[0];

    // Process in chunks of VECTOR_SIZE
    for (int i = 1; i < n; i += VECTOR_SIZE) {
        __m256d phase_vec = _mm256_loadu_pd(&phases[i]);
        __m256d prev_unwrapped_vec = _mm256_loadu_pd(&phases[i-1]);

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

void fftfreq(double *freqs) {
    double n = 1024;
    double d = 1.0;
    int half_n = (n + 1) / 2;  // This is the number of positive frequencies (including zero frequency)

    // Compute positive frequencies
    for (int i = 0; i < half_n; i++) {
        freqs[i] = i / (n * d);
    }

    // Compute negative frequencies
    for (int i = half_n; i < n; i++) {
        freqs[i] = (i - n) / (n * d);
    }
}

double avx_dot_product(double *a, double *b, int n) {
    __m256d sum_vec = _mm256_setzero_pd();  // Initialize the sum vector to zero

    int i;
    for (i = 0; i <= n - 4; i += 4) {
        __m256d vec_a = _mm256_loadu_pd(&a[i]);  // Load 4 elements from array a
        __m256d vec_b = _mm256_loadu_pd(&b[i]);  // Load 4 elements from array b
        __m256d vec_mul = _mm256_mul_pd(vec_a, vec_b);  // Perform element-wise multiplication
        sum_vec = _mm256_add_pd(sum_vec, vec_mul);  // Accumulate the results
    }

    // Horizontal addition to sum all elements in sum_vec
    double sum_array[4];
    _mm256_storeu_pd(sum_array, sum_vec);
    double sum = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3];

    // Handle the remaining elements if n is not a multiple of 4
    for (; i < n; i++) {
        sum += a[i] * b[i];
    }

    return sum;
}

// Function to roll the array using AVX
void avx_roll(double *arr, int n, int shift) {
    // Normalize shift to be within the array bounds
    shift = shift % n;
    if (shift < 0) {
        shift += n;
    }

    double *temp = (double*)aligned_alloc(32, sizeof(double) * n);

    // Handle the portion of the array that wraps around
    int wrap_start = n - shift;

    // Use AVX to copy the wrapped around part
    for (int i = 0; i < shift; i += 4) {
        __m256d v = _mm256_loadu_pd(&arr[wrap_start + i]);
        _mm256_storeu_pd(&temp[i], v);
    }

    // Use AVX to copy the remaining part
    for (int i = 0; i < wrap_start; i += 4) {
        __m256d v = _mm256_loadu_pd(&arr[i]);
        _mm256_storeu_pd(&temp[shift + i], v);
    }

    // Copy the result back to the original array
    for (int i = 0; i < n; i += 4) {
        __m256d v = _mm256_loadu_pd(&temp[i]);
        _mm256_storeu_pd(&arr[i], v);
    }

    free(temp);
}

void avx_abs(double *arr, double *result, int n) {
    __m256d sign_mask = _mm256_set1_pd(-0.0); // Sign mask to clear the sign bit

    int i;
    for (i = 0; i <= n - 4; i += 4) {
        __m256d vec = _mm256_loadu_pd(&arr[i]);  // Load 4 elements from array
        __m256d abs_vec = _mm256_andnot_pd(sign_mask, vec); // Clear the sign bit
        _mm256_storeu_pd(&result[i], abs_vec);   // Store the result
    }

    // Handle the remaining elements if n is not a multiple of 4
    for (; i < n; i++) {
        result[i] = fabs(arr[i]);
    }
}
int argmax(cplx *arr, int n) {
    int max_index = 0;
    double max_value = creal(arr[0]);

    for (int i = 1; i < n; i++) {
        if (creal(arr[i]) > max_value) {
            max_value = creal(arr[i]);
            max_index = i;
        }
    }

    return max_index;
}
    

// Function to print an array
void print_array(double* array, int size) {
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



// Function to calculate the mean of differences between two arrays using AVX
double avx_mean_difference(double *arr1, double *arr2, int n) {
    __m256d sum_diff_vec = _mm256_setzero_pd();  // Initialize a vector to accumulate the differences

    int i;
    for (i = 0; i <= n - 4; i += 4) {
        __m256d vec1 = _mm256_loadu_pd(&arr1[i]); // Load 4 elements from arr1
        __m256d vec2 = _mm256_loadu_pd(&arr2[i]); // Load 4 elements from arr2
        __m256d diff_vec = _mm256_sub_pd(vec1, vec2); // Subtract arr2 from arr1
        sum_diff_vec = _mm256_add_pd(sum_diff_vec, diff_vec); // Accumulate the differences
    }

    // Sum the elements in the vector sum_diff_vec
    double sum_diff[4];
    _mm256_storeu_pd(sum_diff, sum_diff_vec);
    double total_sum_diff = sum_diff[0] + sum_diff[1] + sum_diff[2] + sum_diff[3];

    // Handle the remaining elements if n is not a multiple of 4
    for (; i < n; i++) {
        total_sum_diff += (arr1[i] - arr2[i]);
    }

    // Calculate and return the mean of the differences
    return total_sum_diff / n;
}


void hilbert_transform(double *signal1,double *signal2,double *dot) {
    // print_array(signal1,N);
    // print_array(signal2,N);

    cplx *transformed1 = malloc(BUFFER_SIZE * sizeof(cplx));
    cplx *transformed2 = malloc(BUFFER_SIZE * sizeof(cplx));

    cplx *X1 = malloc(BUFFER_SIZE * sizeof(cplx));
    cplx *X2 = malloc(BUFFER_SIZE * sizeof(cplx));

    cplx *analytic_signal1 = malloc(BUFFER_SIZE * sizeof(cplx));
    cplx *analytic_signal2 = malloc(BUFFER_SIZE * sizeof(cplx));


    double *phase1= malloc(BUFFER_SIZE * sizeof(double));
    double *phase2= malloc(BUFFER_SIZE * sizeof(double));


    cplx *H = malloc(BUFFER_SIZE * sizeof(cplx));

    fftw_plan plan1;
    fftw_plan plan2;

    double *unwrapped_phase1 = malloc(BUFFER_SIZE * sizeof(double));
    double *unwrapped_phase2 = malloc(BUFFER_SIZE * sizeof(double));
    double *means_phase_diff = malloc(sizeof(double));
    double *freqs = (double *)malloc(BUFFER_SIZE * sizeof(double));
    int *shift = malloc(sizeof(int));


    for (int i = 0; i < BUFFER_SIZE; i++) {
        if (i == 0 || i == BUFFER_SIZE / 2) {
            H[i] = 1;
        } else if (i < BUFFER_SIZE / 2) {
            H[i] = 2;
        } else {
            H[i] = 0;
        }
    }

    for (int i = 0; i < BUFFER_SIZE; i++) {
        X1[i] = signal1[i];
        X2[i] = signal2[i];
    }



    // Create a plan for the FFT
    plan1 = fftw_plan_dft_1d(BUFFER_SIZE, X1, transformed1, FFTW_FORWARD, FFTW_ESTIMATE);
    plan2 = fftw_plan_dft_1d(BUFFER_SIZE, X2, transformed2, FFTW_FORWARD, FFTW_ESTIMATE);





    // Execute the FFT
    fftw_execute(plan1);
    fftw_execute(plan2);

    for (int i = 0; i < BUFFER_SIZE; i++) {
        X1[i] = transformed1[i] * H[i];   // ??? replace with x 
        X2[i] = transformed2[i] * H[i];
    }


    plan1 = fftw_plan_dft_1d(BUFFER_SIZE, X1, analytic_signal1, FFTW_BACKWARD, FFTW_ESTIMATE);
    plan2 = fftw_plan_dft_1d(BUFFER_SIZE, X2, analytic_signal2, FFTW_BACKWARD, FFTW_ESTIMATE);

    // Execute the IFFT
    fftw_execute(plan1);
    fftw_execute(plan2);


    // BUFFER_SIZEormalize the result
    for (int i = 0; i < BUFFER_SIZE; ++i) {
        analytic_signal1[i] /= BUFFER_SIZE; // BUFFER_SIZEormalize the result
        analytic_signal2[i] /= BUFFER_SIZE; // BUFFER_SIZEormalize the result
    }
    // Destroy the FFT plan
    fftw_destroy_plan(plan1);
    fftw_destroy_plan(plan2);

    // compute_ifft(X_filtered, analytic_signal, BUFFER_SIZE);




    
    for (int i = 0; i < BUFFER_SIZE; ++i) {
        phase1[i] = carg(analytic_signal1[i]);
        phase2[i] = carg(analytic_signal2[i]);

    }

    // print_array(phase1,BUFFER_SIZE);
    




    unwrap_phase2(phase1,unwrapped_phase1,BUFFER_SIZE);
    unwrap_phase2(phase2,unwrapped_phase2,BUFFER_SIZE);

    // print_array(unwrapped_phase1,BUFFER_SIZE);
    // print_array(unwrapped_phase2,BUFFER_SIZE);


    *means_phase_diff = avx_mean_difference(unwrapped_phase1,unwrapped_phase2,BUFFER_SIZE);


    // double freqs[BUFFER_SIZE];

    
    // Call the fftfreq function
    fftfreq(freqs);
  


    *shift = (int)round(*means_phase_diff*sample_rate/freq);

    // printf("shift = %d\n",shift);

    roll(signal1,BUFFER_SIZE,*shift);

    *dot = 0;

    for (int i = 0; i < BUFFER_SIZE; i++)
    {
        *dot +=signal1[i] * signal2[i];
    }
    //    printf("%f\n",dot);
    
    // avx_roll(signal1,BUFFER_SIZE,shift);

    // double dot = avx_dot_product(signal1,signal2,BUFFER_SIZE);

    // printf("%f\n",dot);
    // double fft_abs1[BUFFER_SIZE];

    // for (int i = 0; i < BUFFER_SIZE; i++)
    // {
    //     fft_abs1[i] = fabs(creal(transformed1[i]));
    // }
    // printf("%d\n",argmax(fft_abs1,BUFFER_SIZE));

    // avx_abs(fft_real1,fft_abs1,BUFFER_SIZE);
    // means_phase_diff = 0.0 ; // (double)argmax(fft_abs1,BUFFER_SIZE);

    // // round();


    free(X1);
    free(X2);
    free(analytic_signal1);
    free(analytic_signal2);
    free(transformed1);
    free(transformed2);
    free(phase1);
    free(phase2);
    free(unwrapped_phase1);
    free(unwrapped_phase2);
    free(freqs);
    free(means_phase_diff);
    free(shift);
    free(H);














}




void *calculate_phase_difference(void *arg) {
    connection_info *conn_info1 = ((connection_info **)arg)[0];
    connection_info *conn_info2 = ((connection_info **)arg)[1];
    FILE *file = fopen("phase_difference.txt", "w");
    if (!file) {
        perror("File opening failed");
        return NULL;
    }

    double *dot = malloc(sizeof(double));;
    double sampling_rate = 1024.0;  // Adjust according to actual sampling rate

    // int i = 0; 
    while (1) {
        // i += 1; 
        pthread_mutex_lock(&conn_info1->buffer_mutex);
        pthread_mutex_lock(&conn_info2->buffer_mutex);




        // double unwrapped_phase1[BUFFER_SIZE];
        // double unwrapped_phase2[BUFFER_SIZE];

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