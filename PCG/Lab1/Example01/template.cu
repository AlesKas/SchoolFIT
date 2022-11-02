 /**
 * @file      template.cu
 *
 * @author    Jiri Jaros \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            jarosjir@fit.vutbr.cz
 *
 * @brief     PCG - PC lab 1 / Print out CUDA thread and block id
 *
 * @version   2021
 *
 * @date      07 October   2020, 15:49 (created) \n
 * @date      07 October   2020, 16:53 (created) \n
 *
 */

#include <cstdio>

 //----------------------------------------------------------------------------//
 //                             Helper functions                               //
 //----------------------------------------------------------------------------//

 /**
  * Check last error -  This routine check the error from the CUDA calls.
  */
void cudaCheckError()
{
    cudaError_t e = cudaGetLastError();

    if (e != cudaSuccess)
    {
        printf("Cuda failure '%s'\n", cudaGetErrorString(e));
        exit(EXIT_FAILURE);
    }
}// end of cudaCheckError
//-----------------------------------------------------------------------------


////////////////////////////////////////////////////////////////////////////////
//                       Implement following functions                        //
////////////////////////////////////////////////////////////////////////////////

/**
 * What to do here:
 *   1. Change this routine to a CUDA kernel.
 *   2. Get the id of the current block (bx) and thread (tx).
 */
__global__ void cudaHelloWorld(void)
{
    // Find the block and thread ID.
    int bx = blockIdx.x;
    int tx = threadIdx.x;

    printf("Hello world. I'm block = %d, thread = %d.\n", bx, tx);
}// end of cudaHelloWorld
//------------------------------------------------------------------------------

/**
 * What to do here:
 *   1. Call the cudaHellWorld kernel on the GPU with 2 blocks and 5 threads
 *      per block.
 *   2. Try to remove CUDA device synchronize and watch what happens.
 */
int main()
{
    // Change this routine call to CUDA kernel call
    cudaHelloWorld<<<32,32>>>();

    // Try to comment this routine and watch what happens.
    cudaDeviceSynchronize();


    cudaCheckError();

    return 0;
}// end of main
//------------------------------------------------------------------------------