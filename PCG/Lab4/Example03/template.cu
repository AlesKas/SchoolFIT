/**
 * @file      template.cpp
 *
 * @author    Jiri Jaros \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            jarosjir@fit.vutbr.cz
 *
 * @brief     PC lab 4 / Reduction
 *
 * @version   2021
 *
 * @date      06 November  2020, 10:02 (created) \n
 * @date      11 November  2020, 10:03 (revised) \n
 *
 */
#include <wb.h>

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                       Function to be implemented                                                   //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * Calculate the sum of the vector elements.
 * @param [in] output - Output value
 * @param [in] input  - Input array
 * @param [in] size   - Size of the input array
 */
__global__ void cudaReduction(int*       output,
                              const int* input,
                              const int  size)
{
  // Load a segment of the input vector into shared memory
  extern __shared__ int partialSum[];


  // Clear partial sum
  partialSum[threadIdx.x] = 0;
  __syncthreads();

  // Reduce the array into partial sums
  int threadId = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = threadId; i < size; i += gridDim.x * blockDim.x) {
    partialSum[threadIdx.x] += input[i]; 
  }
  __syncthreads();


  // Reduction kernel
  // Traverse the reduction tree.
  for (int stride = blockDim.x >> 1; stride >= 1; stride >>= 1) {
    if (threadIdx.x < stride) {
      partialSum[threadIdx.x] += partialSum[threadIdx.x + stride];
    }
    __syncthreads();
  }



  // Write the computed sum of the block to the output vector at the
  if (threadIdx.x == 0) {
    atomicAdd(output, partialSum[0]);
  }


}// end of cudaReduction
//----------------------------------------------------------------------------------------------------------------------

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * Main routine - complete necessary tasks.
 * @param argc
 * @param argv
 * @return
 */
int main(int argc, char **argv)
{
  // Number of elements in the input list
  int inputLength;

  // Parse command line.
  wbArg_t args = wbArg_read(argc, argv);

  int* hostInput  = (int *) wbImport(wbArg_getInputFile(args, 0), &inputLength, "Integer");
  // Only one element
  int* hostOutput = (int *) malloc(sizeof(int));

  printf("----------------- Example 3: CUDA reduction in shared memory -----------------\n");
  printf("Input size: %d\n", inputLength);

  printf("Input sequence: ");
  for (int i = 0 ; i < inputLength ; i++)
  {
    printf("%d, ",hostInput[i]);
  }
  printf("\n");

  int* deviceInput;
  int* deviceOutput;
  cudaMalloc<int>(&deviceInput,  sizeof(int) * inputLength);
  cudaMalloc<int>(&deviceOutput, sizeof(int));


  //Copy memory to the GPU
  cudaMemcpy(deviceInput, hostInput, sizeof(int) * inputLength, cudaMemcpyHostToDevice);
  cudaMemset(deviceOutput, sizeof(int), 0);

  // Initialize the grid and block dimensions here
  dim3 dimGrid(64, 1, 1);
  dim3 dimBlock(256, 1, 1);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, 0);

  // Launch the GPU Kernel here
  cudaReduction<<<dimGrid, dimBlock, 256 * sizeof(int)>>>(deviceOutput, deviceInput, inputLength);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  float time = 0;
  cudaEventElapsedTime(&time, start, stop);

  printf("Time to reduce in shared memory: %4.2f ms\n", time);

  // Copy the GPU memory back to the CPU here
  cudaMemcpy(hostOutput, deviceOutput, sizeof(int), cudaMemcpyDeviceToHost);
  printf("Final sum: %d\n", hostOutput[0]);

  // Free the GPU memory here
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  float sol = float(hostOutput[0]);
  wbSolution(args, &sol, 1);

  free(hostInput);
  free(hostOutput);

  return 0;
}// end of main
//----------------------------------------------------------------------------------------------------------------------
