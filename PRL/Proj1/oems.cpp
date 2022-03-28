#include <mpi.h>
#include <fstream>
#include <iostream>
#include <algorithm>

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

        for (int i = 0; i < 4; i++) {
            senderBuffer[0] = file.get();
            senderBuffer[1] = file.get(); 
            MPI_Send(senderBuffer, 2, MPI_INT, i, 0, MPI_COMM_WORLD);
            std::cout << senderBuffer[0] << " " << senderBuffer[1] << " ";
        }
        std::cout << std::endl;
    }

    int *buffer;
    buffer = (int*)malloc(sizeof(int*) * 8);

    if (rank == 0) {
        MPI_Recv(buffer, 2, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        if (buffer[0] < buffer[1]) {
            MPI_Send(&buffer[0], 1, MPI_INT, 4, 0, MPI_COMM_WORLD);
            MPI_Send(&buffer[1], 1, MPI_INT, 5, 0, MPI_COMM_WORLD);
        } else {
            MPI_Send(&buffer[1], 1, MPI_INT, 4, 0, MPI_COMM_WORLD);
            MPI_Send(&buffer[0], 1, MPI_INT, 5, 0, MPI_COMM_WORLD);
        }
    }

    if (rank == 1) {
        MPI_Recv(buffer, 2, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        if (buffer[0] < buffer[1]) {
            MPI_Send(&buffer[0], 1, MPI_INT, 4, 1, MPI_COMM_WORLD);
            MPI_Send(&buffer[1], 1, MPI_INT, 5, 1, MPI_COMM_WORLD);
        } else {
            MPI_Send(&buffer[1], 1, MPI_INT, 4, 1, MPI_COMM_WORLD);
            MPI_Send(&buffer[0], 1, MPI_INT, 5, 1, MPI_COMM_WORLD);
        }
    }

    if (rank == 2) {
        MPI_Recv(buffer, 2, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        if (buffer[0] < buffer[1]) {
            MPI_Send(&buffer[0], 1, MPI_INT, 6, 2, MPI_COMM_WORLD);
            MPI_Send(&buffer[1], 1, MPI_INT, 7, 2, MPI_COMM_WORLD);
        } else {
            MPI_Send(&buffer[1], 1, MPI_INT, 6, 2, MPI_COMM_WORLD);
            MPI_Send(&buffer[0], 1, MPI_INT, 7, 2, MPI_COMM_WORLD);
        }
    }

    if (rank == 3) {
        MPI_Recv(buffer, 2, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        if (buffer[0] < buffer[1]) {
            MPI_Send(&buffer[0], 1, MPI_INT, 6, 3, MPI_COMM_WORLD);
            MPI_Send(&buffer[1], 1, MPI_INT, 7, 3, MPI_COMM_WORLD);
        } else {
            MPI_Send(&buffer[1], 1, MPI_INT, 6, 3, MPI_COMM_WORLD);
            MPI_Send(&buffer[0], 1, MPI_INT, 7, 3, MPI_COMM_WORLD);
        }
    }

    if (rank == 4) {
        MPI_Recv(&buffer[0], 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(&buffer[1], 1, MPI_INT, 1, 1, MPI_COMM_WORLD, &status);
        if (buffer[0] < buffer[1]) {
            MPI_Send(&buffer[0], 1, MPI_INT, 10, 4, MPI_COMM_WORLD);
            MPI_Send(&buffer[1], 1, MPI_INT, 8, 4, MPI_COMM_WORLD);
        } else {
            MPI_Send(&buffer[1], 1, MPI_INT, 10, 4, MPI_COMM_WORLD);
            MPI_Send(&buffer[0], 1, MPI_INT, 8, 4, MPI_COMM_WORLD);
        }
    }

    if (rank == 5) {
        MPI_Recv(&buffer[0], 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(&buffer[1], 1, MPI_INT, 1, 1, MPI_COMM_WORLD, &status);
        if (buffer[0] < buffer[1]) {
            MPI_Send(&buffer[0], 1, MPI_INT, 8, 5, MPI_COMM_WORLD);
            MPI_Send(&buffer[1], 1, MPI_INT, 13, 5, MPI_COMM_WORLD);
        } else {
            MPI_Send(&buffer[1], 1, MPI_INT, 8, 5, MPI_COMM_WORLD);
            MPI_Send(&buffer[0], 1, MPI_INT, 13, 5, MPI_COMM_WORLD);
        }
    }

    if (rank == 6) {
        MPI_Recv(&buffer[0], 1, MPI_INT, 2, 2, MPI_COMM_WORLD, &status);
        MPI_Recv(&buffer[1], 1, MPI_INT, 3, 3, MPI_COMM_WORLD, &status);
        if (buffer[0] < buffer[1]) {
            MPI_Send(&buffer[0], 1, MPI_INT, 10, 6, MPI_COMM_WORLD);
            MPI_Send(&buffer[1], 1, MPI_INT, 9, 6, MPI_COMM_WORLD);
        } else {
            MPI_Send(&buffer[1], 1, MPI_INT, 10, 6, MPI_COMM_WORLD);
            MPI_Send(&buffer[0], 1, MPI_INT, 9, 6, MPI_COMM_WORLD);
        }
    }

    if (rank == 7) {
        MPI_Recv(&buffer[0], 1, MPI_INT, 2, 2, MPI_COMM_WORLD, &status);
        MPI_Recv(&buffer[1], 1, MPI_INT, 3, 3, MPI_COMM_WORLD, &status);
        if (buffer[0] < buffer[1]) {
            MPI_Send(&buffer[0], 1, MPI_INT, 9, 7, MPI_COMM_WORLD);
            MPI_Send(&buffer[1], 1, MPI_INT, 13, 7, MPI_COMM_WORLD);
        } else {
            MPI_Send(&buffer[1], 1, MPI_INT, 9, 7, MPI_COMM_WORLD);
            MPI_Send(&buffer[0], 1, MPI_INT, 13, 7, MPI_COMM_WORLD);
        }
    }

    if (rank == 8) {
        MPI_Recv(&buffer[0], 1, MPI_INT, 4, 4, MPI_COMM_WORLD, &status);
        MPI_Recv(&buffer[1], 1, MPI_INT, 5, 5, MPI_COMM_WORLD, &status);
        if (buffer[0] < buffer[1]) {
            MPI_Send(&buffer[0], 1, MPI_INT, 12, 8, MPI_COMM_WORLD);
            MPI_Send(&buffer[1], 1, MPI_INT, 11, 8, MPI_COMM_WORLD);
        } else {
            MPI_Send(&buffer[1], 1, MPI_INT, 12, 8, MPI_COMM_WORLD);
            MPI_Send(&buffer[0], 1, MPI_INT, 11, 8, MPI_COMM_WORLD);
        }
    }

    if (rank == 9) {
        MPI_Recv(&buffer[0], 1, MPI_INT, 6, 6, MPI_COMM_WORLD, &status);
        MPI_Recv(&buffer[1], 1, MPI_INT, 7, 7, MPI_COMM_WORLD, &status);
        if (buffer[0] < buffer[1]) {
            MPI_Send(&buffer[0], 1, MPI_INT, 12, 9, MPI_COMM_WORLD);
            MPI_Send(&buffer[1], 1, MPI_INT, 11, 9, MPI_COMM_WORLD);
        } else {
            MPI_Send(&buffer[1], 1, MPI_INT, 12, 9, MPI_COMM_WORLD);
            MPI_Send(&buffer[0], 1, MPI_INT, 11, 9, MPI_COMM_WORLD);
        }
    }

    if (rank == 10) {
        MPI_Recv(&buffer[0], 1, MPI_INT, 4, 4, MPI_COMM_WORLD, &status);
        MPI_Recv(&buffer[1], 1, MPI_INT, 6, 6, MPI_COMM_WORLD, &status);
        if (buffer[0] < buffer[1]) {
            MPI_Send(&buffer[0], 1, MPI_INT, 0, 10, MPI_COMM_WORLD);
            MPI_Send(&buffer[1], 1, MPI_INT, 14, 10, MPI_COMM_WORLD);
        } else {
            MPI_Send(&buffer[1], 1, MPI_INT, 0, 10, MPI_COMM_WORLD);
            MPI_Send(&buffer[0], 1, MPI_INT, 14, 10, MPI_COMM_WORLD);
        }
    }


    if (rank == 11) {
        MPI_Recv(&buffer[0], 1, MPI_INT, 8, 8, MPI_COMM_WORLD, &status);
        MPI_Recv(&buffer[1], 1, MPI_INT, 9, 9, MPI_COMM_WORLD, &status);
        if (buffer[0] < buffer[1]) {
            MPI_Send(&buffer[0], 1, MPI_INT, 14, 11, MPI_COMM_WORLD);
            MPI_Send(&buffer[1], 1, MPI_INT, 18, 11, MPI_COMM_WORLD);
        } else {
            MPI_Send(&buffer[1], 1, MPI_INT, 14, 11, MPI_COMM_WORLD);
            MPI_Send(&buffer[0], 1, MPI_INT, 18, 11, MPI_COMM_WORLD);
        }
    }


    if (rank == 12) {
        MPI_Recv(&buffer[0], 1, MPI_INT, 8, 8, MPI_COMM_WORLD, &status);
        MPI_Recv(&buffer[1], 1, MPI_INT, 9, 9, MPI_COMM_WORLD, &status);
        if (buffer[0] < buffer[1]) {
            MPI_Send(&buffer[0], 1, MPI_INT, 16, 12, MPI_COMM_WORLD);
            MPI_Send(&buffer[1], 1, MPI_INT, 15, 12, MPI_COMM_WORLD);
        } else {
            MPI_Send(&buffer[1], 1, MPI_INT, 16, 12, MPI_COMM_WORLD);
            MPI_Send(&buffer[0], 1, MPI_INT, 15, 12, MPI_COMM_WORLD);
        }
    }

    if (rank == 13) {
        MPI_Recv(&buffer[0], 1, MPI_INT, 5, 5, MPI_COMM_WORLD, &status);
        MPI_Recv(&buffer[1], 1, MPI_INT, 7, 7, MPI_COMM_WORLD, &status);
        if (buffer[0] < buffer[1]) {
            MPI_Send(&buffer[0], 1, MPI_INT, 15, 13, MPI_COMM_WORLD);
            MPI_Send(&buffer[1], 1, MPI_INT, 0, 13, MPI_COMM_WORLD);
        } else {
            MPI_Send(&buffer[1], 1, MPI_INT, 15, 13, MPI_COMM_WORLD);
            MPI_Send(&buffer[0], 1, MPI_INT, 0, 13, MPI_COMM_WORLD);
        }   
    }

    if (rank == 14) {
        MPI_Recv(&buffer[0], 1, MPI_INT, 10, 10, MPI_COMM_WORLD, &status);
        MPI_Recv(&buffer[1], 1, MPI_INT, 11, 11, MPI_COMM_WORLD, &status);
        if (buffer[0] < buffer[1]) {
            MPI_Send(&buffer[0], 1, MPI_INT, 16, 14, MPI_COMM_WORLD);
            MPI_Send(&buffer[1], 1, MPI_INT, 17, 14, MPI_COMM_WORLD);
        } else {
            MPI_Send(&buffer[1], 1, MPI_INT, 16, 14, MPI_COMM_WORLD);
            MPI_Send(&buffer[0], 1, MPI_INT, 17, 14, MPI_COMM_WORLD);
        }
    }

    if (rank == 15) {
        MPI_Recv(&buffer[0], 1, MPI_INT, 12, 12, MPI_COMM_WORLD, &status);
        MPI_Recv(&buffer[1], 1, MPI_INT, 13, 13, MPI_COMM_WORLD, &status);
        if (buffer[0] < buffer[1]) {
            MPI_Send(&buffer[0], 1, MPI_INT, 17, 15, MPI_COMM_WORLD);
            MPI_Send(&buffer[1], 1, MPI_INT, 18, 15, MPI_COMM_WORLD);
        } else {
            MPI_Send(&buffer[1], 1, MPI_INT, 15, 15, MPI_COMM_WORLD);
            MPI_Send(&buffer[0], 1, MPI_INT, 18, 15, MPI_COMM_WORLD);
        }
    }

    if (rank == 16) {
        MPI_Recv(&buffer[0], 1, MPI_INT, 14, 14, MPI_COMM_WORLD, &status);
        MPI_Recv(&buffer[1], 1, MPI_INT, 12, 12, MPI_COMM_WORLD, &status);
        if (buffer[0] < buffer[1]) {
            MPI_Send(&buffer[0], 1, MPI_INT, 0, 16, MPI_COMM_WORLD);
            MPI_Send(&buffer[1], 1, MPI_INT, 0, 16, MPI_COMM_WORLD);
        } else {
            MPI_Send(&buffer[1], 1, MPI_INT, 0, 16, MPI_COMM_WORLD);
            MPI_Send(&buffer[0], 1, MPI_INT, 0, 16, MPI_COMM_WORLD);
        }   
    }

    if (rank == 17) {
        MPI_Recv(&buffer[0], 1, MPI_INT, 14, 14, MPI_COMM_WORLD, &status);
        MPI_Recv(&buffer[1], 1, MPI_INT, 15, 15, MPI_COMM_WORLD, &status);
        if (buffer[0] < buffer[1]) {
            MPI_Send(&buffer[0], 1, MPI_INT, 0, 17, MPI_COMM_WORLD);
            MPI_Send(&buffer[1], 1, MPI_INT, 0, 17, MPI_COMM_WORLD);
        } else {
            MPI_Send(&buffer[1], 1, MPI_INT, 0, 17, MPI_COMM_WORLD);
            MPI_Send(&buffer[0], 1, MPI_INT, 0, 17, MPI_COMM_WORLD);
        }
    }

    if (rank == 18) {
        MPI_Recv(&buffer[0], 1, MPI_INT, 11, 11, MPI_COMM_WORLD, &status);
        MPI_Recv(&buffer[1], 1, MPI_INT, 15, 15, MPI_COMM_WORLD, &status); 
        if (buffer[0] < buffer[1]) {
            MPI_Send(&buffer[0], 1, MPI_INT, 0, 18, MPI_COMM_WORLD);
            MPI_Send(&buffer[1], 1, MPI_INT, 0, 18, MPI_COMM_WORLD);
        } else {
            MPI_Send(&buffer[1], 1, MPI_INT, 0, 18, MPI_COMM_WORLD);
            MPI_Send(&buffer[0], 1, MPI_INT, 0, 18, MPI_COMM_WORLD);
        }  
    }
    
    if (rank == 0) {
        MPI_Recv(&buffer[0], 1, MPI_INT, 10, 10, MPI_COMM_WORLD, &status);
        MPI_Recv(&buffer[1], 1, MPI_INT, 16, 16, MPI_COMM_WORLD, &status);
        MPI_Recv(&buffer[2], 1, MPI_INT, 16, 16, MPI_COMM_WORLD, &status);
        MPI_Recv(&buffer[3], 1, MPI_INT, 17, 17, MPI_COMM_WORLD, &status);
        MPI_Recv(&buffer[4], 1, MPI_INT, 17, 17, MPI_COMM_WORLD, &status);
        MPI_Recv(&buffer[5], 1, MPI_INT, 18, 18, MPI_COMM_WORLD, &status);
        MPI_Recv(&buffer[6], 1, MPI_INT, 18, 18, MPI_COMM_WORLD, &status);
        MPI_Recv(&buffer[7], 1, MPI_INT, 13, 13, MPI_COMM_WORLD, &status);
        for (int i = 0; i < 8; i++) {
            std::cout <<buffer[i] << std::endl;
        }
    }

    MPI_Finalize(); 
    return 0;
}