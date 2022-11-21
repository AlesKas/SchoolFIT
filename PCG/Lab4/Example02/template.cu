/**
 * @file      solution.cpp
 *
 * @author    Jiri Jaros \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            jarosjir@fit.vutbr.cz
 *
 * @brief     PC lab 4 / Histogram
 *
 * @version   2021
 *
 * @date      06 November  2020, 10:02 (created) \n
 * @date      06 November  2020, 13:06 (revised) \n
 *
 */

#include <wb.h>

#include <string.h>

using std::string;

//--------------------------------------------------------------------------------------------------------------------//
// Useful links:                                                                                                      //
//   i. CUDA SDK documentation https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html                       //
//   ii. List of CUDA routines https://docs.nvidia.com/cuda/cuda-runtime-api/index.html                               //
//--------------------------------------------------------------------------------------------------------------------//

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                       Function to be implemented                                                   //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


//--------------------------------------------------------------------------------------------------------------------//
// 1. Implement the CUDA kernel for calculating histogram using shared memory                                         //
//--------------------------------------------------------------------------------------------------------------------//

/**
 * CUDA kernel to compute histogram in shared memory.
 * @param [out] histogram   - Histogram
 * @param [in]  input       - Input text
 * @param [in]  numBins     - Number of bins
 * @param [in]  inputLenght - Size of the input text
 */
 __global__ void cudaHistogramShared(unsigned int* histogram,
                                     unsigned int  numBins,
                                     const char*   input,
                                     unsigned int  inputLenght)
{
  // Privatized histogram in shared memory.
  // Since we don't know the size of the histogram in the compile time, we have to use extern shared mem without size []



  // Thread Id and stride (grid size).



  // Clear private histogram - Consider a possibility the histogram can be bigger than the block size.



  // Traverse through the text and store the histogram into shared memory.
  // If there's no bin for a given char, use the last one



  // Add this part to global histogram.



}// end of cudaHistogramShared
//----------------------------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------------------------//
// 2. Implement the routine for calculating histogram in shared memory                                                //
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Compute histogram using shared memory.
 * @param [out] histogram   - Histogram
 * @param [in]  input       - Input text
 * @param [in]  numBins     - Number of bins
 * @param [in]  inputLenght - Size of the input text
 */
void histogramShared(unsigned int* histogram,
                     unsigned int  numBins,
                     const char*   input,
                     unsigned int  inputLenght)
{
  // Zero the histogram in global memory.



  // Launch histogram kernel on the bins
  dim3 blockDim(256), gridDim(64);
  // The third parameter of the starting configuration will be the size of the SM.



}// end of histogramShared
//----------------------------------------------------------------------------------------------------------------------

/**
 * Main routine - complete necessary tasks.
 * @param argc
 * @param argv
 * @return
 */
int main(int argc, char *argv[])
{
  // Parse command line.
  wbArg_t args = wbArg_read(argc, argv);

  // Number of bins in the histogram
  constexpr unsigned int nBins = 128;
  // Input length
  int inputLength = 0;

  // Read input files
  char*          hostInput      = (char *) wbImport(wbArg_getInputFile(args, 0), &inputLength, "Text");
  unsigned int*  hostHistogram  = (unsigned int *) malloc(nBins * sizeof(unsigned int));

  printf("----------------- Example 2: CUDA histogram in shared memory -----------------\n");
  printf("Input size:     %d\n", inputLength);
  printf("Histogram size: %d\n", nBins);


  // Allocate device memory for histogram and input data
  char*         deviceInput;
  unsigned int* deviceHistogram;

  cudaMalloc<char>(&deviceInput, inputLength * sizeof(char));
  cudaMalloc<unsigned int>(&deviceHistogram, nBins * sizeof(unsigned int));

  // Copy input data to GPU
  cudaMemcpy(deviceInput, hostInput, inputLength, cudaMemcpyHostToDevice);

  // Shared histogram
  cudaEvent_t start, stop;

  //------------------------------------------------ Shared histogram ------------------------------------------------//

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, 0);
  // Launch kernel
  histogramShared(deviceHistogram, nBins, deviceInput, inputLength);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  float time = 0;
  cudaEventElapsedTime(&time, start, stop);

  printf("Time to histogram in shared memory:  %4.2f ms\n", time);

  // Copy the GPU memory back to the CPU here
  cudaMemcpy(hostHistogram, deviceHistogram, nBins * sizeof(unsigned int), cudaMemcpyDeviceToHost);
  wbSolution(args, hostHistogram, nBins);

  cudaFree(deviceInput);
  cudaFree(deviceHistogram);

  free(hostHistogram);
  free(hostInput);

  return 0;
}// end of main
//----------------------------------------------------------------------------------------------------------------------
