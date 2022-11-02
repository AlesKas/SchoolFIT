/**
 * @file      template.cu
 *
 * @author    Jiri Jaros \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            jarosjir@fit.vutbr.cz
 *
 * @brief     PC lab 1 / Vector addition
 *
 * @version   2021
 *
 * @date      08 October   2020, 10:27 (created) \n
 * @date      08 October   2022, 22:08 (revised) \n
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
 * CUDA kernel to add two vectors out[i] = in1[i] + in2[i]
 * @param [out] out  - Output vector
 * @param [in]  in1  - First input vector
 * @param [in]  in2  - Second input vector
 * @param [in]  size - Size of the vectors
 */

__global__ void cudaAddVectors(float*       out,
                               const float* in1,
                               const float* in2,
                               const int    size)
{
  // Insert code to implement vector addition here
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < size) {
      out[i] = in1[i] + in2[i];
    }
}// end of cudaAddVectors
//----------------------------------------------------------------------------------------------------------------------

/**
 * Main routine - complete necessary tasks.
 * @param argc
 * @param argv
 * @return
 */
int main(int argc, char **argv)
{
  // Array size
  int inputLength;

  // Read commandline arguments
  wbArg_t args = wbArg_read(argc, argv);

  // Import data from the files
  float* hostInput1 = (float *) wbImport(wbArg_getInputFile(args, 0), &inputLength);
  float* hostInput2 = (float *) wbImport(wbArg_getInputFile(args, 1), &inputLength);
  float* hostOutput = (float *) malloc(inputLength * sizeof(float));

  printf("---------------- Example 2: CUDA vector addition ----------------\n");
  printf("Number of elements: %d\n", inputLength);


  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //                                              Tasks to be implemented                                             //
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  // Device vectors
  float* deviceInput1 = nullptr;
  float* deviceInput2 = nullptr;
  float* deviceOutput = nullptr;

  //------------------------------------------------------------------------------------------------------------------//
  // 1. Allocate GPU memory here (cudaMalloc)                                                                         //
  //------------------------------------------------------------------------------------------------------------------//
  cudaMalloc(&deviceInput1, inputLength * sizeof(float));
  cudaMalloc(&deviceInput2, inputLength * sizeof(float));
  cudaMalloc(&deviceOutput, inputLength * sizeof(float));


  //------------------------------------------------------------------------------------------------------------------//
  // 2. Copy both input vectors on the GPU (cudaMemcpy)                                                               //
  // 3. Clear output vectors on GPU by     (cudaMemset)                                                               //
  //------------------------------------------------------------------------------------------------------------------//
  cudaMemcpy(deviceInput1, hostInput1, inputLength * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceInput2, hostInput2, inputLength * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemset(deviceOutput, 0, inputLength * sizeof(float));

  //------------------------------------------------------------------------------------------------------------------//
  // 4. Initialize the grid and block dimensions                                                                      //
  //------------------------------------------------------------------------------------------------------------------//
  int threadsPerBlock = 256;
  int blocksPerGrid = (inputLength + threadsPerBlock - 1) / threadsPerBlock;


  //------------------------------------------------------------------------------------------------------------------//
  // 5. Launch the GPU Kernel here                                                                                    //
  //------------------------------------------------------------------------------------------------------------------//
  cudaAddVectors<<<blocksPerGrid, threadsPerBlock>>>(deviceOutput, deviceInput1, deviceInput2, inputLength);


  //------------------------------------------------------------------------------------------------------------------//
  // 6. Copy the GPU memory back to the CPU here (cudaMemcpy)                                                         //
  //------------------------------------------------------------------------------------------------------------------//
  cudaMemcpy(hostOutput, deviceOutput, inputLength * sizeof(float), cudaMemcpyDeviceToHost);
  // for (int i = 0; i < inputLength; i++) {
  //   printf("%f = %f + %f\n", hostOutput[i], hostInput1[i], hostInput2[i]);
  // }

  //------------------------------------------------------------------------------------------------------------------//
  // 7. Free the GPU memory here (cudaFree)                                                                           //
  //------------------------------------------------------------------------------------------------------------------//
  cudaFree(hostInput1);
  cudaFree(hostInput2);
  cudaFree(hostOutput);

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  wbSolution(args, hostOutput, inputLength);

  // Free host data.
  free(hostInput1);
  free(hostInput2);
  free(hostOutput);

  return 0;
}// end of main
//----------------------------------------------------------------------------------------------------------------------
