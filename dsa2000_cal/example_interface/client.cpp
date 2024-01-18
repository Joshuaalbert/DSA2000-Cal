// client.cpp
#include <iostream>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <cstring>

int main() {
    const int array_size = 10;
    const int port = 12345;
    const char* array_ready = "ARRAY_READY";
    const char* server_processing = "SERVER_PROCESSING";
    const char* client_processing = "CLIENT_PROCESSING";
    const char* client_completed = "COMPLETED";



    int* shared_data = nullptr;
    int shm_fd = -1;

    // Socket setup
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        perror("Socket creation failed");
        return 1;
    }

    struct sockaddr_in serv_addr;
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(port);
    serv_addr.sin_addr.s_addr = INADDR_ANY;

    if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
        perror("Connection failed");
        close(sock);
        return 1;
    }

    char event_buffer[1025] = {0};

    // Get ready event
    int read_bytes = read(sock, event_buffer, 1024);
    if (read_bytes < 0) {
        perror("Read failed");
        close(sock);
        return 1;
    }
    event_buffer[read_bytes] = '\0';
    std::cout << "Message from server: " << event_buffer << std::endl;

    if (strcmp(event_buffer, array_ready) != 0) {
        // Wrong event
        perror("Wrong event from server.");
    } else {
        std::cout << "Array ready to attach to." << std::endl;
        char shm_name[33] = {0};

        // Connect to array
        read_bytes = read(sock, shm_name, 32);
        if (read_bytes < 0) {
            perror("Read failed");
            close(sock);
            return 1;
        }

        shm_name[read_bytes] = '\0'; // Null-terminate the received message
        std::cout << "Received shared memory name: " << shm_name << std::endl;

        shm_fd = shm_open(shm_name, O_RDWR, 0666);
        if (shm_fd == -1) {
            perror("shm_open");
            close(sock);
            return 1;
        }

        shared_data = (int*)mmap(NULL, sizeof(int) * array_size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
        if (shared_data == MAP_FAILED) {
            perror("mmap");
            close(shm_fd);
            close(sock);
            return 1;
        }

        // Init array
        for (int i = 0; i < array_size; ++i) {
            shared_data[i] = 2;
        }
        std::cout << "Initialised" << std::endl;

    }

    int count = 0;
    while (count < 5) {

        // Tell server to do something
        std::cout << "Server's turn" << std::endl;
        if (send(sock, server_processing, strlen(server_processing), 0) < 0) {
            perror("Send failed");
        }

        // Wait until client's turn
        //
        // Waiting until External system does its part.
        //
        int read_bytes = read(sock, event_buffer, 1024);
        if (read_bytes < 0) {
            perror("Read failed");
            close(sock);
            return 1;
        }
        event_buffer[read_bytes] = '\0';
        std::cout << "Message from server: " << event_buffer << std::endl;

        if (strcmp(event_buffer, client_processing) != 0) {
            // Wrong event
            perror("Wrong event from server.");
        } else {
            std::cout << "Client processing array." << std::endl;
            // Modify shared memory
            //
            // This would be syscam's turn to work on array.
            //
            for (int i = 0; i < array_size; ++i) {
                shared_data[i] += 2;
            }
        }
        count++;
    }

    // Say we're done
    std::cout << "Letting server know we're done" << std::endl;
    if (send(sock, client_completed, strlen(client_completed), 0) < 0) {
        perror("Send failed");
    }


    // Cleanup
    if (shared_data != nullptr) {
        munmap(shared_data, sizeof(int) * array_size);
    }
    if (shm_fd != -1) {
        close(shm_fd);
    }
    close(sock);
    return 0;
}
