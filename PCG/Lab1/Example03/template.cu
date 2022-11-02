/**
 * @file      template.cu
 *
 * @author    Jiri Jaros \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            jarosjir@fit.vutbr.cz
 *
 * @brief     PC lab 1 / Covert RGB image to levels of gray
 *
 * @version   2021
 *
 * @date      08 October   2020, 20:17 (created) \n
 * @date      13 October   2022, 22:20 (revised) \n
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
 * Convert RBG image into a gray one using 2D kernel
 * The formula to convert is:
 *
 *  grayImage[i] = 0.21f * rgbImage[3 * i] + 0.71f *  rgbImage[3 * i + 1] + 0.07f *  rgbImage[3 * i + 2];
 *
 * @param [out] grayImage - Output grey image
 * @param [in]  rgbImage  - Input RBG image
 * @param [in]  channels  - Number of color channels
 * @param [in]  height    - Number of rows (y size)
 * @param [in]  width     - Number of cols (x size)
 */
 __global__ void rgb2gray(float*       grayImage,
                          const float* rgbImage,
                          const int    channels,
                          const int    height,
                          const int    width)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x < width && y < height) {
    int grayOffset = y * width + x;
    int rgbOffset = grayOffset * channels;
    const float r = rgbImage[rgbOffset];
    const float g = rgbImage[rgbOffset + 1];
    const float b = rgbImage[rgbOffset + 2];
    grayImage[grayOffset] = 0.21f * r + 0.71f * g + 0.07f * b;
  }
}// end of rgb2gray
//----------------------------------------------------------------------------------------------------------------------

/**
 * Main routine - complete necessary tasks.
 * @param argc
 * @param argv
 * @return
 */
int main(int argc, char *argv[])
{
  // Data to be allocated on the GPU.
  float *deviceInputImageData;
  float *deviceOutputImageData;

  // Parse commandline parameters
  wbArg_t args = wbArg_read(argc, argv);

  // Read input file
  wbImage_t inputImage = wbImport(wbArg_getInputFile(args, 0));

  // Get image sizes
  const int imageWidth  = wbImage_getWidth(inputImage);
  const int imageHeight = wbImage_getHeight(inputImage);
  // For this lab the value is always 3
  const int imageChannels = wbImage_getChannels(inputImage);

  // Since the image is monochromatic, it only contains one channel
  wbImage_t outputImage = wbImage_new(imageWidth, imageHeight, 1);

  // Get image data and covert it to floats (faster execution on GPUs)
  float* hostInputImageData  = wbImage_getData(inputImage);
  float* hostOutputImageData = wbImage_getData(outputImage);

  printf("---------------- Example 3: Image conversion from RGB to gray----------------\n");
  printf("Image size [%d, %d]\n", imageWidth, imageHeight);


  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //                                              Tasks to be implemented                                             //
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


  //------------------------------------------------------------------------------------------------------------------//
  // 1. Allocate GPU memory here (cudaMalloc)                                                                         //
  //------------------------------------------------------------------------------------------------------------------//
  int inputSize = imageWidth * imageHeight * imageChannels;
  cudaMalloc<float>(&deviceInputImageData, inputSize * sizeof(float));
  cudaMalloc<float>(&deviceOutputImageData, imageWidth * imageHeight * sizeof(float));


  //------------------------------------------------------------------------------------------------------------------//
  // 2. Copy both input vectors on the GPU (cudaMemcpy)                                                               //
  // 3. Clear output vectors on GPU by     (cudaMemset)                                                               //
  //------------------------------------------------------------------------------------------------------------------//
  cudaMemcpy(deviceInputImageData, hostInputImageData, inputSize * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemset(deviceOutputImageData, 0, imageWidth * imageHeight * sizeof(float));


  //------------------------------------------------------------------------------------------------------------------//
  // 4. Initialize the grid and block dimensions, use 2D kernel                                                       //
  //------------------------------------------------------------------------------------------------------------------//

  constexpr int TILE_WIDTH = 16;
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
  dim3 dimGrid(ceil(float(imageWidth) / TILE_WIDTH),
               ceil(float(imageHeight) / TILE_WIDTH)  
  );

  //------------------------------------------------------------------------------------------------------------------//
  // 5. Launch the GPU Kernel here                                                                                    //
  //------------------------------------------------------------------------------------------------------------------//
  rgb2gray<<<dimGrid, dimBlock>>>(deviceOutputImageData, deviceInputImageData, imageChannels, imageHeight, imageWidth);


  //------------------------------------------------------------------------------------------------------------------//
  // 6. Copy the GPU memory back to the CPU here (cudaMemcpy)                                                         //
  //------------------------------------------------------------------------------------------------------------------//
  cudaMemcpy(hostOutputImageData, deviceOutputImageData, imageWidth * imageHeight * sizeof(float), cudaMemcpyDeviceToHost);


  //------------------------------------------------------------------------------------------------------------------//
  // 7. Free the GPU memory here (cudaFree)                                                                           //
  //------------------------------------------------------------------------------------------------------------------//
  cudaFree(deviceOutputImageData);
  cudaFree(hostOutputImageData);


  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  wbSolution(args, outputImage);

  wbImage_delete(outputImage);
  wbImage_delete(inputImage);

  return 0;
}// end of main
//----------------------------------------------------------------------------------------------------------------------