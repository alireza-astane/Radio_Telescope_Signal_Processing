#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#define PORT 8080 // Port to listen on
#define BUFFER_SIZE 1024 // Size of buffer to read data

int main() {
    int server_socket, client_socket;
    struct sockaddr_in server_address, client_address;
    socklen_t client_address_size;
    char buffer[BUFFER_SIZE];
    int bytes_received;

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

    printf("Server listening on port %d...\n", PORT);

    // Accept a client connection
    client_address_size = sizeof(client_address);
    client_socket = accept(server_socket, (struct sockaddr *)&client_address, &client_address_size);
    if (client_socket == -1) {

perror("Accept error");
        close(server_socket);
        exit(1);
    }

    printf("Client connected from %s:%d\n", inet_ntoa(client_address.sin_addr), ntohs(client_address.sin_port));

    // Receive data from the client
    FILE *fp = fopen("received_data.bin", "wb"); // Open file for writing binary data
    if (fp == NULL) {
        perror("Error opening file");
        exit(1);
    }

    while ((bytes_received = recv(client_socket, buffer, BUFFER_SIZE, 0)) > 0) {
        fwrite(buffer, 1, bytes_received, fp); // Write received data to file
    }

    fclose(fp);

    // Close sockets
    close(client_socket);
    close(server_socket);

    printf("Data received and saved to received_data.bin\n");

    return 0;
}