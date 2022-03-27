#include <mpi.h>
#include <fstream>
#include <iostream>
#include <algorithm>

int merge(int *bufferData, int numsPerFirst, int *remote, int numsPerSecond, int *merged) {
    int i,j;
    int index=0;

    for (i=0,j=0; i < numsPerFirst; i++) {
        while ((remote[j] < bufferData[i]) && j < numsPerSecond) {
            merged[index++] = remote[j++];
        }
        merged[index++] = bufferData[i];
    }
    while (j < numsPerSecond)
        merged[index++] = remote[j++];

    return 0;
}

void ExchangePairs(int numsPerProc, int *bufferData, int sendrank, int recvrank, MPI_Comm comm) {
    int rank;
    int remote[numsPerProc];
    int merged[2*numsPerProc];
    const int mergetag = 1;
    const int sortedtag = 2;

    MPI_Comm_rank(comm, &rank);
    if (rank == sendrank) {
        MPI_Send(bufferData, numsPerProc, MPI_INT, recvrank, mergetag, MPI_COMM_WORLD);
        MPI_Recv(bufferData, numsPerProc, MPI_INT, recvrank, sortedtag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else {
        MPI_Recv(remote, numsPerProc, MPI_INT, sendrank, mergetag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        merge(bufferData, numsPerProc, remote, numsPerProc, merged);

        int theirstart = 0, mystart = numsPerProc;
        if (sendrank > rank) {
            theirstart = numsPerProc;
            mystart = 0;
        }
        MPI_Send(&(merged[theirstart]), numsPerProc, MPI_INT, sendrank, sortedtag, MPI_COMM_WORLD);
        for (int i=mystart; i<mystart+numsPerProc; i++)
            bufferData[i-mystart] = merged[i];
    }
}

int main(int argc, char *argv[]) {
    std::string inputFile = "numbers";
    std::streampos fileSize;
    std::ifstream file(inputFile, std::ios::binary);

    file.seekg(0, std::ios::end);
    fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    int rank, ranks;
	int *data;
    int bufferData[1];
    int numsPerProc;
	MPI_Status status;

    data = (int*) malloc(sizeof(int*) * fileSize);
    memset(data, 0, sizeof(int*) * fileSize);
	
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    if(rank == 0) {
        int index = 0;
        int number;
        numsPerProc = fileSize / ranks;
        while (file.good()) {
            number = file.get();
            if(!file.good()) 
                break; 
            data[index] = number;
            std::cout << data[index] << " ";
            index++;
        }
        std::cout << std::endl;
    }
    else{
    	data=NULL;
    }

    MPI_Bcast(&numsPerProc, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(data, numsPerProc, MPI_INT, &bufferData, 1, MPI_INT, 0, MPI_COMM_WORLD);
    for (int i = 1; i <= fileSize; i++) {
        if ((i + rank) % 2 == 0) {
            if (rank < ranks - 1) {
                ExchangePairs(numsPerProc, bufferData, rank, rank + 1, MPI_COMM_WORLD);
            }
        } else if (rank > 0) {
            ExchangePairs(numsPerProc, bufferData, rank - 1, rank , MPI_COMM_WORLD);
        }
    }
    MPI_Gather(bufferData, numsPerProc, MPI_INT, data, numsPerProc, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        for (int i = 0; i < fileSize; i++) {
            std::cout << data[i] << std::endl;
        }
    }

    MPI_Finalize(); 
    return 0;
}