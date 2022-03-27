#include <mpi.h>
#include <fstream>
#include <iostream>
#include <algorithm>

// int merge(int *bufferData, int numsPerFirst, int *remote, int numsPerSecond, int *merged) {
//     int i,j;
//     int index=0;

//     for (i=0,j=0; i < numsPerFirst; i++) {
//         while ((remote[j] < bufferData[i]) && j < numsPerSecond) {
//             merged[index++] = remote[j++];
//         }
//         merged[index++] = bufferData[i];
//     }
//     while (j < numsPerSecond)
//         merged[index++] = remote[j++];

//     return 0;
// }

int main(int argc, char *argv[]) {
    std::string inputFile = "numbers";
    std::streampos fileSize;
    std::ifstream file(inputFile, std::ios::binary);

    file.seekg(0, std::ios::end);
    fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    int rank, ranks;
	MPI_Status status;

	
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    if(rank == 0) {

        int *senderBuffer = (int*) malloc(sizeof(int*) * 2);
        memset(senderBuffer, 0, sizeof(int*) * 2);
        senderBuffer[0] = file.get();
        senderBuffer[1] = file.get(); 
        MPI_Send(senderBuffer, 2, MPI_INT, 1, 0, MPI_COMM_WORLD);
        std::cout << senderBuffer[0] << " " << senderBuffer[1] << " ";

        senderBuffer[0] = file.get();
        senderBuffer[1] = file.get();
        MPI_Send(senderBuffer, 2, MPI_INT, 2, 0, MPI_COMM_WORLD);
        std::cout << senderBuffer[0] << " " << senderBuffer[1] << " ";

        senderBuffer[0] = file.get();
        senderBuffer[1] = file.get();
        MPI_Send(senderBuffer, 2, MPI_INT, 3, 0, MPI_COMM_WORLD);
        std::cout << senderBuffer[0] << " " << senderBuffer[1] << " ";

        senderBuffer[0] = file.get();
        senderBuffer[1] = file.get();
        MPI_Send(senderBuffer, 2, MPI_INT, 4, 0, MPI_COMM_WORLD);
        std::cout << senderBuffer[0] << " " << senderBuffer[1] << std::endl;
    }


    int *buffer;
    buffer = (int*)malloc(sizeof(int*) * 8);

    if (rank == 1) {
        MPI_Recv(buffer, 2, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        std:: cout << "[" << rank << "] recieved: ";
        for (int i = 0; i < 2; i++) {
            std::cout << buffer[i] << " ";
        }
        std::cout << std::endl;
    }

    if (rank == 2) {
        MPI_Recv(buffer, 2, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        std:: cout << "[" << rank << "] recieved: ";
        for (int i = 0; i < 2; i++) {
            std::cout << buffer[i] << " ";
        }
        std::cout << std::endl;
    }

    if (rank == 3) {
        MPI_Recv(buffer, 2, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        std:: cout << "[" << rank << "] recieved: ";
        for (int i = 0; i < 2; i++) {
            std::cout << buffer[i] << " ";
        }
        std::cout << std::endl;
    }

    if (rank == 4) {
        MPI_Recv(buffer, 2, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        std:: cout << "[" << rank << "] recieved: ";
        for (int i = 0; i < 2; i++) {
            std::cout << buffer[i] << " ";
        }
        std::cout << std::endl;
    }

    MPI_Finalize(); 
    return 0;
}