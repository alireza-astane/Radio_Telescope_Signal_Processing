#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <math.h>
#include <pthread.h>

#define SERVER_PORT 12345
#define BUFFER_SIZE 1024
#define PI 3.14159265358979323846

double sine_wave[BUFFER_SIZE];
double frequency = 2.0; // 1 Hz
double amplitude = 1.0; // Amplitude of the sine wave
double sampling_rate = 1.0; // Sampling rate in Hz (adjust as needed)

void generate_sine_wave(double *buffer, int length, double frequency, double amplitude, double sampling_rate) {
    for (int i = 0; i < length; i++) {
        buffer[i] = amplitude * sin(frequency * i / sampling_rate);
    }
}

void *send_data(void *arg) {
    int client_fd = *(int *)arg;

    while (1) {
        generate_sine_wave(sine_wave, BUFFER_SIZE, frequency, amplitude, sampling_rate);
        if (send(client_fd, sine_wave, sizeof(sine_wave), 0) < 0) {
            perror("Send failed");
            break;
        }
        usleep(10000 / sampling_rate * BUFFER_SIZE); // Sleep to control the data sending rate
    }

    close(client_fd);
    return NULL;
}

int main() {
    int server_fd, client_fd;
    struct sockaddr_in server_addr, client_addr;
    socklen_t client_addr_len = sizeof(client_addr);

    // Create socket
    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
        perror("Socket creation failed");
        exit(EXIT_FAILURE);
    }

    // Setup server address
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(SERVER_PORT);

    // Bind socket to the address
    if (bind(server_fd, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
        perror("Bind failed");
        close(server_fd);
        exit(EXIT_FAILURE);
    }

    // Listen for incoming connections
    if (listen(server_fd, 3) < 0) {
        perror("Listen failed");
        close(server_fd);
        exit(EXIT_FAILURE);
    }

    printf("Server listening on port %d\n", SERVER_PORT);

    // Accept a client connection
    if ((client_fd = accept(server_fd, (struct sockaddr *)&client_addr, &client_addr_len)) < 0) {
        perror("Accept failed");
        close(server_fd);
        exit(EXIT_FAILURE);
    }

    printf("Client connected\n");

    // Create a thread to send data
    pthread_t thread_id;
    pthread_create(&thread_id, NULL, send_data, &client_fd);

    // Wait for the thread to finish
    pthread_join(thread_id, NULL);

    // Close the server socket
    close(server_fd);

    return 0;
}
