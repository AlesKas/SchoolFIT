/**
 * @file      template.cpp
 *
 * @author    Jiri Jaros \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            jarosjir@fit.vutbr.cz
 *
 * @brief     PC lab 2 / Matrix multiplication - naive version
 *
 * @version   2021
 *
 * @date      17 October   2020, 15:11 (created) \n
 * @date      20 October   2020, 16:09 (created) \n
 *
 */

#include <wb.h>

//--------------------------------------------------------------------------------------------------------------------//
// Useful links:                                                                                                      //
//   i. CUDA SDK documentation https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html                       //
//   ii. List of CUDA routines https://docs.nvidia.com/cuda/cuda-runtime-api/index.html                               //
//--------------------------------------------------------------------------------------------------------------------//

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                       Function to be implemented                                                   //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * CUDA kernel to compute single precision matrix-matrix multiplications (sgemm)
 * Compute C[i][j] = sum_i(sum_j(sum_k(A[i][k] * B[k][j])));
 *
 * @param [out] C - Output matrix  [N * P]
 * @param [in]  A - Input matrix A [N * M]
 * @param [in]  B - Input matrix B [M * K]
 * @param [in]  N - Number of A's rows
 * @param [in]  M - Number of A's cols and B's rows
 * @param [in]  P - Number of B's rows
 */
__global__ void cudaMatrixMatrixMul(float*       C,
                                    const float* A,
                                    const float* B,
                                    const int    N,
                                    const int    M,
                                    const int    P)
{
  // Insert code to implement vector addition here
  // Use a 2D kernel where each thread calculates one element of the C matrix.

   // C[i][j] = sum_i( sum_j( sum_k(A[i][k] * B[k][j])));
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;

  if ((row < N) && (col < P))
  {
    // Store the sum in a register
    float sum = 0.f;

    for (int i = 0; i < M; i++)
    {
      sum += A[row * M + i] * B[i * P + col];
    }

    C[row * P + col] = sum;
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
  // Matrix sizes A[N * M], B [M * P]
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

  printf("---------------- Example 1: CUDA naive matrix-matrix multiplication ----------------\n");
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
  // Block dim
  dim3 blockDim(32, 32);
  // Grid dim
  dim3 gridDim(ceil(float(P) / blockDim.x),
               ceil(float(N) / blockDim.y));



  //------------------------------------------------------------------------------------------------------------------//
  // 5. Launch the GPU Kernel here                                                                                    //
  //------------------------------------------------------------------------------------------------------------------//
  cudaEventRecord(start, 0);
  cudaMatrixMatrixMul<<<gridDim, blockDim>>>(deviceC, deviceA, deviceB, N, M, P);


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
