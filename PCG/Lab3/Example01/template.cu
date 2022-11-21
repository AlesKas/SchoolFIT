/**
 * @file      template.cpp
 *
 * @author    Jiri Jaros \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            jarosjir@fit.vutbr.cz
 *
 * @brief     PC lab 3 / Texture memory with data in global memory
 *
 * @details   This sample takes an input PGM image (image_filename) and generates an output PGM image
 *            (image_filename_out).  This CUDA kernel performs a simple 2D transform (rotation) on the texture
 *            coordinates (u,v).
 *
 * @version   2022
 *
 * @date      27 October   2020, 11:36 (created) \n
 * @date      27 October   2022, 10:02 (revised by Ondrej Olsak)
 *
 */

//--------------------------------------------------------------------------------------------------------------------//
// Useful links:                                                                                                      //
//   i.   CUDA SDK documentation: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html                    //
//   ii.  List of CUDA routines:  https://docs.nvidia.com/cuda/cuda-runtime-api/index.html                            //
//   iii. Texture memory:         https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-memory     //
//--------------------------------------------------------------------------------------------------------------------//

// Includes, system
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <math.h>

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

// Includes CUDA
#include <cuda_runtime.h>
// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h
// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check


/// Define the files that are to be save and the reference images for validation
const char* imageFilename = "lena.pgm";
const char* refFilename   = "lena_ref.pgm";

/// Max acceptable error
constexpr float makEpsilonError = 5e-3f;
// angle to rotate image by (in radians)
constexpr float rotationAngle   = 0.5f;


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                       Functions to be implemented                                                  //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * Transform an image using texture lookups
 * @param [out] outputImage - Output image
 * @param [in]  texImage    - Cuda texture object
 * @param [in]  height      - Image height
 * @param [in]  width       - Image width
 * @param [in]  float       - Rotation angle
 */
__global__ void transformKernel(float*      outputImage,
                                cudaTextureObject_t texImage,
                                const int   height,
                                const int   width,
                                const float theta)
{
  // Get position of the thread in the image
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  // Read from texture and write to global memory
  if (y < height && x < width)
  {
    // Normalized coordinates <0,1>
    float u = x / float(width);
    float v = y / float(height);

    // Move coordinates to the middle of the image
    u -= 0.5f;
    v -= 0.5f;

    // Calculate rotated coordinates
    const float tu = u * cosf(theta) - v * sinf(theta) + 0.5f;
    const float tv = v * cosf(theta) + u * sinf(theta) + 0.5f;

    //----------------------------------------------------------------------------------------------------------------//
    // 1. Write pixel [tu, tv] into outputImage                                                                       //
    //----------------------------------------------------------------------------------------------------------------//


    //----------------------------------------------------------------------------------------------------------------//
  }
} // end of transformKernel
//----------------------------------------------------------------------------------------------------------------------

/**
 * Run a simple test for CUDA
 */
bool runTest(int argc, char** argv)
{
  char* imagePath = sdkFindFilePath(imageFilename, argv[0]);

  if (imagePath == nullptr)
  {
    printf("Unable to source image file: %s\n", imageFilename);
    exit(EXIT_FAILURE);
  }

  // Image size
  unsigned int width  = 0;
  unsigned int height = 0;
  // load image from disk
  float* hImage = nullptr;

  sdkLoadPGM(imagePath, &hImage, &width, &height);

  const int size = width * height;
  printf("Loaded '%s', %d x %d pixels\n", imageFilename, width, height);

  //  Load reference image from image (output)
  float* hRefImage    = (float *) malloc(size * sizeof(float));
  // Allocate mem for the result on host side
  float* hOutputImage = (float *) malloc(size * sizeof(float));

  char*  refPath      = sdkFindFilePath(refFilename, argv[0]);

  if (refPath == nullptr)
  {
    printf("Unable to find reference image file: %s\n", refFilename);
    exit(EXIT_FAILURE);
  }

  sdkLoadPGM(refPath, &hRefImage, &width, &height);


  // Time measurement
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //                                              Tasks to be implemented                                             //
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  size_t pitch;

  float* dOutputImage = nullptr;
  float* dInputImage  = nullptr;

  // Allocate GPU memory for input and output image here
  cudaMallocPitch<float>(&dInputImage, &pitch, width*sizeof(float), height);
  cudaMalloc<float>(&dOutputImage, size * sizeof(float));

  // Copy input image on the GPU
  cudaMemcpy2D(dInputImage, pitch, hImage, width*sizeof(float), width*sizeof(float),
               height, cudaMemcpyHostToDevice);

  //------------------------------------------------------------------------------------------------------------------//
  // 2. Create Channel Format                                                                                         //
  //------------------------------------------------------------------------------------------------------------------//

  // Create Texture descriptor
  cudaChannelFormatDesc channelDesc;
  // use cudaCrateChannelDesc
  // Lecture no.5, slide 30
  channelDesc = ;

  //------------------------------------------------------------------------------------------------------------------//
  // 4. Set texture parameters                                                                                        //
  //    https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaTextureDesc.html#structcudaTextureDesc
  //------------------------------------------------------------------------------------------------------------------//
  struct cudaTextureDesc textureDesc;
  memset(&textureDesc, 0, sizeof(textureDesc));

  // cudaAddressModeWrap        Wrapping address mode
  // cudaAddressModeClamp       Clamp to edge address mode
  // cudaAddressModeMirror      Mirror address mode
  // cudaAddressModeBorder      Border address mode
  textureDesc.addressMode[0] = ; // ?
  textureDesc.addressMode[1] = ; // ?

  // cudaFilterModePoint  Point filter mode
  // cudaFilterModeLinear Linear filter mode
  textureDesc.filterMode = ; //?

  // access with normalized texture coordinates
  textureDesc.normalizedCoords = ; //?

  //------------------------------------------------------------------------------------------------------------------//
  // 5. Set resource descriptor
  //    https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaResourceDesc.html#structcudaResourceDesc
  //------------------------------------------------------------------------------------------------------------------//
  struct cudaResourceDesc resourceDesc;
  memset(&resourceDesc, 0, sizeof(resourceDesc));

  resourceDesc.resType = ;
  resourceDesc.res.pitch2D.devPtr = ;
  resourceDesc.res.pitch2D.desc = ;
  resourceDesc.res.pitch2D.width  = ;
  resourceDesc.res.pitch2D.height = ;
  resourceDesc.res.pitch2D.pitchInBytes = ;

  //------------------------------------------------------------------------------------------------------------------//
  // 6. Create texture object                                                                                //
  //------------------------------------------------------------------------------------------------------------------//

  cudaTextureObject_t texImage;

  //------------------------------------------------------------------------------------------------------------------//

  // Initialize the grid and block dimensions
  dim3 dimBlock(16, 16);
  dim3 dimGrid(width / dimBlock.x, height / dimBlock.y);

  // Launch the GPU kernel
  cudaEventRecord(start, 0);
  // Execute the kernel
  transformKernel<<<dimGrid, dimBlock, 0>>>(dOutputImage, texImage, height, width, rotationAngle);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  float elapsedTime = 0;
  cudaEventElapsedTime(&elapsedTime, start, stop);

  printf("Processing time:     %f (ms)\n", elapsedTime);
  printf("Processing perf: %.2f Mpixels/sec\n", (width * height / (elapsedTime / 1000.0f)) / 1e6);

  // Copy result from device to host
  cudaMemcpy(hOutputImage, dOutputImage, size * sizeof(float), cudaMemcpyDeviceToHost);

  //------------------------------------------------------------------------------------------------------------------//
  // 7. Unbind texture and free memory                                                                               //
  //------------------------------------------------------------------------------------------------------------------//


  //------------------------------------------------------------------------------------------------------------------//

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  cudaFree(dOutputImage);
  cudaFree(dInputImage);

  // Write result to file
  char outputFilename[1024];
  strcpy(outputFilename, imagePath);
  strcpy(outputFilename + strlen(imagePath) - 4, "_out.pgm");
  sdkSavePGM(outputFilename, hOutputImage, width, height);
  printf("Wrote '%s'\n", outputFilename);

  bool testResult = false;
  // Write regression file if necessary
  if (checkCmdLineFlag(argc, (const char **) argv, "regression"))
  {
    // Write file for regression test
    sdkWriteFile<float>("./data/regression.dat", hOutputImage, width * height,  0.0f, false);
  }
  else
  {
    // We need to reload the data from disk,
    // because it is inverted upon output
    sdkLoadPGM(outputFilename, &hOutputImage, &width, &height);

    printf("Comparing files\n");
    printf("\toutput:    <%s>\n", outputFilename);
    printf("\treference: <%s>\n", refPath);

    testResult = compareData(hOutputImage, hRefImage, width * height, makEpsilonError, 0.15f);
  }

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  free(imagePath);
  free(refPath);
  return testResult;
}// end of runTest
//----------------------------------------------------------------------------------------------------------------------
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 *  Main routine
 * @param argc
 * @param argv
 * @return
 */
int main(int argc, char** argv)
{
  // Process command-line arguments
  if (argc > 1)
  {
    if (checkCmdLineFlag(argc, (const char **) argv, "input"))
    {
      getCmdLineArgumentString(argc, (const char **) argv, "input", (char **) &imageFilename);

      if (checkCmdLineFlag(argc, (const char **) argv, "reference"))
      {
        getCmdLineArgumentString(argc, (const char **) argv, "reference", (char **) &refFilename);
      }
      else
      {
        printf("-input flag should be used with -reference flag\n");
        exit(EXIT_FAILURE);
      }
    }
    else if (checkCmdLineFlag(argc, (const char **) argv, "reference"))
    {
      printf("-reference flag should be used with -input flag\n");
      exit(EXIT_FAILURE);
    }
  }

  const bool testResult = runTest(argc, argv);

  printf("Completed, returned %s\n", testResult ? "OK" : "ERROR!");
  exit(testResult ? EXIT_SUCCESS : EXIT_FAILURE);
}// end of main
//----------------------------------------------------------------------------------------------------------------------
