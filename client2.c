#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <pthread.h>

#define SERVER_PORT 12345
#define BUFFER_SIZE 1024

double buffer[BUFFER_SIZE];
pthread_mutex_t buffer_mutex = PTHREAD_MUTEX_INITIALIZER;

void *receive_data(void *arg) {
    int sock = *(int *)arg;
    int bytes_received;

    while ((bytes_received = recv(sock, buffer, sizeof(buffer), 0)) > 0) {
        pthread_mutex_lock(&buffer_mutex);
        // Process the received data (e.g., store it or analyze it)
        for (int i = 0; i < bytes_received / sizeof(double); i++) {
            printf("%f\n", buffer[i]);
        }
        pthread_mutex_unlock(&buffer_mutex);
    }

    if (bytes_received < 0) {
        perror("Receive failed");
    }

    close(sock);
    return NULL;
}

int main() {
    int sock;
    struct sockaddr_in server_addr;

    // Create socket
    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        perror("Socket creation failed");
        exit(EXIT_FAILURE);
    }

    // Setup server address
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(SERVER_PORT);
    if (inet_pton(AF_INET, "127.0.0.1", &server_addr.sin_addr) <= 0) {
        perror("Invalid address/ Address not supported");
        close(sock);
        exit(EXIT_FAILURE);
    }

    // Connect to server
    if (connect(sock, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
        perror("Connection failed");
        close(sock);
        exit(EXIT_FAILURE);
    }

    printf("Connected to server\n");

    // Create a thread to receive data
    pthread_t thread_id;
    pthread_create(&thread_id, NULL, receive_data, &sock);

    // Wait for the thread to finish
    pthread_join(thread_id, NULL);

    return 0;
}
