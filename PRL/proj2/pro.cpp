#include <mpi.h>
#include <math.h>
#include <iostream>

typedef struct edge {
    int rank;
    int left;
    int right;
} edge;

typedef struct adjacencyList {
    struct edge firstNode; // first node of tree
    struct edge edge;
    struct edge reverseEdge;
    struct adjacencyList *next;
    int rank;
    int eToure;
    int parent;
    int direction;
} adjacencyList;

adjacencyList* createAdjList(char* input, int numNodes, int numEdges, int rank) {
    // Even indexes represent forward edge, odd indexes represent backward edge
    struct edge *edges = (edge*)malloc(sizeof(edge) * numEdges);
    int index = 0;
    bool switchValue = true;
    for (int i = 0; i < numEdges; i += 2) {
        if (switchValue) {
            edges[i].left = index;
            edges[i].right = (2 * index) + 1;
            edges[i].rank = i;

            edges[i+1].left = (2 * index) + 1;
            edges[i+1].right = index;
            edges[i+1].rank = i + 1;
            switchValue = false;
        } else {
            edges[i].left = index;
            edges[i].right = (2 * index) + 2;
            edges[i].rank = i; 

            edges[i+1].left = (2 * index) + 2;
            edges[i+1].right = index;
            edges[i+1].rank = i + 1;            
            index++;    
            switchValue = true;
        }
    }

    struct adjacencyList *adjList = (adjacencyList*)malloc(sizeof(adjacencyList) * numEdges);
    index = 0;
    switchValue = true;
    int listItemIndex = 0;
    for (int i = 0; i < numEdges; i += 2) {
        if (switchValue) {
            adjList[i].rank = listItemIndex;
            adjList[i].edge = edges[i];
            adjList[i].reverseEdge = edges[i+1];
            if (listItemIndex == 0 && numNodes > 2) {
                adjList[i].firstNode = edges[(i/2)-1];
                adjList[i].next = &adjList[2];
            } else if (i + 2 < numEdges) {
                adjList[i].firstNode = edges[(i/2)-1];
                adjList[i].next = &adjList[i+2];
            } else {
                adjList[i].firstNode = edges[(i/2)-1];
                adjList[i].next = NULL;
            }

            adjList[i+1].rank = (2 * listItemIndex) +1;
            adjList[i+1].firstNode = edges[i+1];
            adjList[i+1].edge = edges[i+1];
            adjList[i+1].reverseEdge = edges[i];
            if (i+1 < (numNodes - 2)) {
                adjList[i+1].next = &adjList[2*(i+1)+2];
            } else {
                adjList[i+1].next = NULL;
            }
            switchValue = false;
        } else {
            adjList[i].rank = listItemIndex;
            adjList[i].edge = edges[i];
            adjList[i].reverseEdge = edges[i+1];
            adjList[i].next = NULL;
            if (listItemIndex == 0 && numNodes > 2) {
                adjList[i].firstNode = edges[0];
            } else {
                adjList[i].firstNode = edges[(i/2)-2];
            }

            adjList[i+1].rank = (2 * listItemIndex) + 2;
            adjList[i+1].edge = edges[i+1];
            adjList[i+1].reverseEdge = edges[i];
            adjList[i+1].firstNode = edges[i+1];
            if (i < (numNodes - 2)) {
                adjList[i+1].next = &adjList[2*(i+1)+2];
            } else {
                adjList[i+1].next = NULL;
            }
            listItemIndex++;
            switchValue = true;
        }
    }
    free(edges);
    return adjList;
}

int main(int argc, char *argv[]) {
    int numProcessors;
    int rank;
    MPI_Status status;

    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcessors);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    char* inputString = argv[1];
    int numNodes = atoi(argv[2]);
    int numEdges = numProcessors;
    
    struct adjacencyList *adjList = createAdjList(inputString, numNodes, numEdges, rank);

    // If there is input of size 1, I dont need to compute anything
    if (numNodes == 1) {
        if (rank == 0) {
            std::cout << inputString << std::endl;
        }
        MPI_Finalize();
        return 0;
    }

    // Alg Vytvoření Eulerovy cesty from lecture
    int eToure[numEdges];
    for(int procc = 0; procc < numProcessors; procc++) {
        if(rank == procc) {
            if (adjList[adjList[procc].reverseEdge.rank].next != NULL) {
                eToure[procc] = adjList[adjList[procc].reverseEdge.rank].next->edge.rank;
            } else {
                eToure[procc] = adjList[adjList[procc].reverseEdge.rank].firstNode.rank;
            }

            if (eToure[procc] == 0) {
                eToure[procc] = -1;
            }
            adjList[procc].eToure = eToure[procc];

            if (adjList[procc].edge.rank < adjList[procc].reverseEdge.rank) {
                adjList[procc].direction = 1;
            } else {
                adjList[procc].direction = 0;
            }
        }
    }

    // Alg Predecessor computing
    int successors[numProcessors];
    int predeccessors[numProcessors];
    int predc;
    MPI_Allgather(&adjList[rank].eToure, 1, MPI_INT, successors, 1, MPI_INT, MPI_COMM_WORLD);
    for(int procc = 1; procc < numProcessors; procc++) {
        if(rank == procc) {
            predeccessors[successors[procc]] = procc;
            MPI_Send(&predeccessors[successors[procc]], 1, MPI_INT, 0, 0,  MPI_COMM_WORLD);
        }
        if (rank == 0) {
            MPI_Recv(&predc, 1, MPI_INT, procc, 0, MPI_COMM_WORLD, &status);
            // Special case for first and last elements
            predeccessors[successors[procc]] = predc;
            predeccessors[0]= -1;
            predeccessors[successors[0]]= 0;
        }
    }

    MPI_Bcast(&predeccessors[0], numProcessors, MPI_INT, 0, MPI_COMM_WORLD);
    adjList[rank].parent = predeccessors[rank];

    // Alg sum suffix from lecture
    int suffixSum[numEdges];
    for(int procc = 0; procc < numProcessors; procc++){
        if(rank == procc) {
            if (adjList[procc].eToure == -1) {
                suffixSum[procc] = 0; // neutral element to suffix sum
            } else {
                suffixSum[procc] = adjList[procc].direction;
            }
            int successor;
            int successorsSuccessor;
            int predecessorsPredecessor;
            for (int k = 0; k < log(numProcessors) + 1; k++) {
                if (adjList[rank].parent == -1 && adjList[rank].eToure == -1) {
                    continue;
                } else if (adjList[rank].parent == -1) {
                    MPI_Recv(&successor, 1, MPI_INT, adjList[rank].eToure, 0, MPI_COMM_WORLD, &status);
                    MPI_Recv(&successorsSuccessor, 1, MPI_INT, adjList[rank].eToure, 0, MPI_COMM_WORLD, &status);
                    MPI_Send(&adjList[rank].parent, 1, MPI_INT, adjList[rank].eToure, 0,  MPI_COMM_WORLD);
                    suffixSum[rank] += successor;
                    adjList[rank].eToure = successorsSuccessor;
                } else if (adjList[rank].eToure == -1) {
                    MPI_Send(&suffixSum[rank], 1, MPI_INT, adjList[rank].parent, 0,  MPI_COMM_WORLD);
                    MPI_Send(&adjList[rank].eToure, 1, MPI_INT, adjList[rank].parent, 0,  MPI_COMM_WORLD);
                    MPI_Recv(&predecessorsPredecessor, 1, MPI_INT, adjList[rank].parent, 0, MPI_COMM_WORLD, &status);
                    adjList[rank].parent = predecessorsPredecessor;
                } else {
                    MPI_Send(&suffixSum[rank], 1, MPI_INT, adjList[rank].parent, 0,  MPI_COMM_WORLD);
                    MPI_Send(&adjList[rank].eToure, 1, MPI_INT, adjList[rank].parent, 0,  MPI_COMM_WORLD);
                    MPI_Recv(&successor, 1, MPI_INT, adjList[rank].eToure, 0, MPI_COMM_WORLD, &status);
                    MPI_Recv(&successorsSuccessor, 1, MPI_INT, adjList[rank].eToure, 0, MPI_COMM_WORLD, &status);
                    MPI_Recv(&predecessorsPredecessor, 1, MPI_INT, adjList[rank].parent, 0, MPI_COMM_WORLD, &status);
                    MPI_Send(&adjList[rank].parent, 1, MPI_INT, adjList[rank].eToure, 0,  MPI_COMM_WORLD);
                    suffixSum[procc] += successor;
                    adjList[rank].eToure = successorsSuccessor;
                    adjList[rank].parent = predecessorsPredecessor;                 
                }
            }
            if (adjList[rank].direction == 1) {
                suffixSum[rank] = numNodes-suffixSum[rank]-1;
            } else {
                suffixSum[rank] = -1;
            }
        }
    }

    int preorder[numNodes];
    int position;
    for(int i = 0; i < numProcessors; i++) {
        if(rank == i) {
            MPI_Send(&suffixSum[rank], 1, MPI_INT, 0, 0,  MPI_COMM_WORLD);
        }
        if(rank == 0) {
            MPI_Recv(&position, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &status);
            // This needs to be >=, == exits with seg fault.
            if (position >= 0) {
                preorder[position] = i;
            }
        }
    }

    if (rank == 0) {  
        printf("%c", argv[1][0]);
        for (int i = 0; i < numNodes - 1; i++) {
            printf("%c", inputString[adjList[preorder[i]].edge.right]);
        }
        std::cout << std::endl;
    }
    MPI_Finalize();
    return 0;
}