#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <pthread.h>

#define SERVER_PORT1 12345
#define SERVER_PORT2 12346
#define BUFFER_SIZE 1024

typedef struct {
    int socket;
    double buffer[BUFFER_SIZE];
    pthread_mutex_t buffer_mutex;
} connection_info;

void *receive_data(void *arg) {
    connection_info *conn_info = (connection_info *)arg;
    int bytes_received;

    while ((bytes_received = recv(conn_info->socket, conn_info->buffer, sizeof(conn_info->buffer), 0)) > 0) {
        pthread_mutex_lock(&conn_info->buffer_mutex);
        // Process the received data (e.g., store it or analyze it)
        for (int i = 0; i < bytes_received / sizeof(double); i++) {
            printf("Server %d: %f\n", conn_info->socket, conn_info->buffer[i]);
        }
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

    // Create threads to receive data
    pthread_t thread_id1, thread_id2;
    pthread_create(&thread_id1, NULL, receive_data, &conn_info1);
    pthread_create(&thread_id2, NULL, receive_data, &conn_info2);

    // Wait for the threads to finish
    pthread_join(thread_id1, NULL);
    pthread_join(thread_id2, NULL);

    return 0;
}
