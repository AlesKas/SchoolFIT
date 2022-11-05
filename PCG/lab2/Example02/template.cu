/**
 * @file      template.cpp
 *
 * @author    Jiri Jaros \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            jarosjir@fit.vutbr.cz
 *
 * @brief     PC lab 2 / Matrix multiplication - SM memory
 *
 * @version   2021
 *
 * @date      17 October   2020, 16:27 (created) \n
 * @date      21 October   2020, 10:10 (created) \n
 *
 */

#include <wb.h>

/**
 * CUDA kernel to compute single precision matrix-matrix multiplications (sgemm)
 * Compute C[i][j] = sum_i(sum_j(sum_k(A[i][k] * B[k][j])));
 *
 * @tparam TILE_WIDTH  - Tile width and height in shared memory (square tile)
 * @param [out] C - Output matrix  [N * P]
 * @param [in]  A - Input matrix A [N * M]
 * @param [in]  B - Input matrix B [M * K]
 * @param [in]  N - Number of A's rows
 * @param [in]  M - Number of A's cols and B's rows
 * @param [in]  P - Number of B's rows
 */
template<int TileWidth>
__global__ void cudaMatrixMatrixMul(float*       C,
                                    const float* A,
                                    const float* B,
                                    const int    N,
                                    const int    M,
                                    const int    P)
{
  // 1. Allocate square blocks in shared memory for matrix A and B
  __shared__ float AA[TileWidth][TileWidth];
  __shared__ float BB[TileWidth][TileWidth + 1];

  // 2. Get 2D blocks' and threads' ids
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  // 3. Row and coll in the C matrix
  const int col = bx * TileWidth + tx;
  const int row = by * TileWidth + ty;
  // 4. Initialize one value of P in a register
  float pVal = 0.f;
  // 5. Go over all tiles. Be careful, the M dimension doesn't have to be divisible by TileWidth!
  for (int m = 0; m < (M - 1) / TileWidth + 1; m++) {
    // 6. Load one element of the A matrix into the shared memory. If being out-of-bound, load a value of 0.0f
    //    Use coalesced reading from matrix A
    AA[ty][tx] = ((row < N) && (m * TileWidth + tx < M)) ? A[row * M + m * TileWidth + tx]   : 0.0f;

    // 7. Load one element of the B matrix into the shared memory. If being out-of-bound, load a value of 0.0f
    //    Use coalesced reading from matrix B
    BB[ty][tx] = ((col < P) && (m * TileWidth + ty < M)) ? B[(m * TileWidth + ty) * P + col] : 0.0f;

    // 8. Finish loading by calling a barrier
    __syncthreads();

    // 9. Calculate a vector dot product of an appropriate row and col in the shared memory and accumulate it
    // into a register
    #pragma unroll 4
    for (int k = 0; k < TileWidth; k++) {
      pVal += AA[ty][k] * BB[k][tx];
    }

    // 10. Finish computation of the tile by a barrier
    __syncthreads();
  }

  // 11. If still being within the bounds of the final array, store the value.
  if ((row < N) && (col < P))
  {
    C[row * P + col] = pVal;
  }
}// end of cudaMatrixMatrixMul
//----------------------------------------------------------------------------------------------------------------------

/**
 * Main routine - complete necessary tasks.
 * @param argc
 * @param argv
 * @return
 */
int main(int argc, char** argv)
{
  // Matrix sizes A[N * M], B [M * P], C [N * P]
  int N = 0;
  int M = 0;
  int P = 0;

  // Time measurement
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Read commandline arguments
  wbArg_t args = wbArg_read(argc, argv);

  // Import data from the files
  float* hostA = (float *) wbImport(wbArg_getInputFile(args, 0), &N, &M);
  float* hostB = (float *) wbImport(wbArg_getInputFile(args, 1), &M, &P);

  // Allocate the hostC matrix
  float* hostC = (float *) malloc(N * P * sizeof(float));

  printf("---------------- Example 2: CUDA shared memory matrix-matrix multiplication ----------------\n");
  printf("Matrix A [%d, %d]\n", N, M);
  printf("Matrix B [%d, %d]\n", M, P);
  printf("Matrix C [%d, %d]\n", N, P);
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //                                              Tasks to be implemented                                             //
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  // Device vectors
  float* deviceA;
  float* deviceB;
  float* deviceC;

  //------------------------------------------------------------------------------------------------------------------//
  // 1. Allocate GPU memory here (cudaMalloc)                                                                         //
  //------------------------------------------------------------------------------------------------------------------//
  cudaMalloc<float>(&deviceA, N * M * sizeof(float));
  cudaMalloc<float>(&deviceB, M * P * sizeof(float));
  cudaMalloc<float>(&deviceC, N * P * sizeof(float));
  //------------------------------------------------------------------------------------------------------------------//
  // 2. Copy both input matrices on the GPU (cudaMemcpy)                                                              //
  // 3. Clear output matrix on GPU by       (cudaMemset)                                                              //
  //------------------------------------------------------------------------------------------------------------------//
  cudaMemcpy(deviceA, hostA, N * M * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, M * P * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemset(deviceC, 0, N * P * sizeof(float));
  //------------------------------------------------------------------------------------------------------------------//
  // 4. Initialize the grid and block dimensions                                                                      //
  //------------------------------------------------------------------------------------------------------------------//

  constexpr int TileWidth = 32;
  // Block dim
  dim3 blockDim(TileWidth, TileWidth);
  // Grid dim
  dim3 gridDim(ceil(float(P) / blockDim.x),
               ceil(float(N) / blockDim.y));
  //------------------------------------------------------------------------------------------------------------------//
  // 5. Launch the GPU Kernel here                                                                                    //
  //------------------------------------------------------------------------------------------------------------------//
  cudaEventRecord(start, 0);
  cudaMatrixMatrixMul<TileWidth><<<gridDim, blockDim>>>(deviceC, deviceA, deviceB, N, M, P);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  float time = 0;
  cudaEventElapsedTime(&time, start, stop);

  printf("Time to calculate the answer:  %3.1f ms \n", time);

  //------------------------------------------------------------------------------------------------------------------//
  // 6. Copy the GPU memory back to the CPU here (cudaMemcpy)                                                         //
  //------------------------------------------------------------------------------------------------------------------//
  cudaMemcpy(hostC, deviceC, N * P * sizeof(float), cudaMemcpyDeviceToHost);


  //------------------------------------------------------------------------------------------------------------------//
  // 7. Free the GPU memory here (cudaFree)                                                                           //
  //------------------------------------------------------------------------------------------------------------------//
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);


  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  wbSolution(args, hostC, N, P);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}// end of main
//----------------------------------------------------------------------------------------------------------------------
