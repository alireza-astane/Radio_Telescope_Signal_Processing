#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>

#define SERVER_PORT 12345
#define BUFFER_SIZE 1024

int main() {
    int sock;
    struct sockaddr_in server_addr;
    double buffer[BUFFER_SIZE];
    int bytes_received;

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

    // Receive and print sine wave data continuously
    while ((bytes_received = recv(sock, buffer, sizeof(buffer), 0)) > 0) {
        for (int i = 0; i < bytes_received / sizeof(double); i++) {
            printf("%f\n", buffer[i]);
        }
    }

    if (bytes_received < 0) {
        perror("Receive failed");
    }

    // Close the socket
    close(sock);

    return 0;
}
