#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#define PORT 8080 // Port to listen on
#define SAMPLE_RATE 44100 // Sampling rate (Hz)
#define DURATION 5 // Duration of sine wave (seconds)
#define AMPLITUDE 0.5 // Amplitude of sine wave
#define PI 3.14159265358979323846

int main() {
    // Socket setup
    int server_socket, client_socket;
    struct sockaddr_in server_address, client_address;
    socklen_t client_address_size;

    // Create a socket
    server_socket = socket(AF_INET, SOCK_STREAM, 0);
    if (server_socket == -1) {
        perror("Socket creation error");
        exit(1);
    }

    // Set address information
    memset(&server_address, 0, sizeof(server_address));
    server_address.sin_family = AF_INET;
    server_address.sin_addr.s_addr = INADDR_ANY;
    server_address.sin_port = htons(PORT);

    // Bind the socket to the address
    if (bind(server_socket, (struct sockaddr *)&server_address, sizeof(server_address)) == -1) {
        perror("Bind error");
        close(server_socket);
        exit(1);
    }

    // Listen for incoming connections
    if (listen(server_socket, 5) == -1) {
        perror("Listen error");
        close(server_socket);
        exit(1);
    }

    // Accept a client connection
    client_address_size = sizeof(client_address);
    client_socket = accept(server_socket, (struct sockaddr *)&client_address, &client_address_size);
    if (client_socket == -1) {
        perror("Accept error");
        close(server_socket);
        exit(1);
    }

    printf("Client connected from %s:%d\n", inet_ntoa(client_address.sin_addr), ntohs(client_address.sin_port));

    // Generate sine wave data
    int num_samples = SAMPLE_RATE * DURATION;
    short int *samples = malloc(num_samples * sizeof(short int));
    for (int i = 0; i < num_samples; i++) {
        double angle = 2 * PI * i / SAMPLE_RATE;
        samples[i] = (short int)(AMPLITUDE * 32767 * sin(angle)); // Normalize to 16-bit range
    }

    // Stream the sine wave data
    int bytes_sent;
    for (int i = 0; i < num_samples; i++) {
        bytes_sent = send(client_socket, &samples[i], sizeof(short int), 0);
        if (bytes_sent == -1) {
            perror("Send error");
            break;
        }
    }

    // Close sockets
    close(client_socket);
    close(server_socket);
    free(samples);

    return 0;
}