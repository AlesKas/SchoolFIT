/**
 * @file      main.cpp
 *
 * @author    Ales Kasparek - xkaspa48 \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            xkaspa48@stud.fit.vutbr.cz
 *
 * @brief     PCG Assignment 2
 *            N-Body simulation in ACC
 *
 * @version   2022
 *
 * @date      11 November  2020, 11:22 (created) \n
 * @date      15 November  2022, 14:03 (revised) \n
 *
 */

#include <chrono>
#include <cstdio>
#include <cmath>

#include "nbody.h"
#include "h5Helper.h"

/**
 * Main routine of the project
 * @param argc
 * @param argv
 * @return
 */
int main(int argc, char **argv)
{
  // Parse command line parameters
  if (argc != 7)
  {
    printf("Usage: nbody <N> <dt> <steps> <write intesity> <input> <output>\n");
    exit(EXIT_FAILURE);
  }

  const int   N         = std::stoi(argv[1]);
  const float dt        = std::stof(argv[2]);
  const int   steps     = std::stoi(argv[3]);
  const int   writeFreq = (std::stoi(argv[4]) > 0) ? std::stoi(argv[4]) : 0;

  printf("N: %d\n", N);
  printf("dt: %f\n", dt);
  printf("steps: %d\n", steps);

  const size_t recordsNum = (writeFreq > 0) ? (steps + writeFreq - 1) / writeFreq : 0;


  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //                                         Code to be implemented                                                   //
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  // 1.  Memory allocation on CPU
  Particles particles_1(N);

  // 2. Create memory descriptor
  /*
   * Caution! Create only after CPU side allocation
   * parameters:
   *                                    Stride of two               Offset of the first
   *             Data pointer           consecutive elements        element in floats,
   *                                    in floats, not bytes        not bytes
  */
  MemDesc md(
                   &particles_1.positions[0].x,  4, 0,            // Position in X
                   &particles_1.positions[0].y,  4, 0,            // Position in Y
                   &particles_1.positions[0].z,  4, 0,            // Position in Z
                   &particles_1.velocities[0].x, 3, 0,            // Velocity in X
                   &particles_1.velocities[0].y, 3, 0,            // Velocity in Y
                   &particles_1.velocities[0].z, 3, 0,            // Velocity in Z
                   &particles_1.positions[0].w, 4, 0,            // Weight
                   N,                                           // Number of particles
                   recordsNum);                                 // Number of records in output file



  H5Helper h5Helper(argv[5], argv[6], md);

  // Read data
  try
  {
    h5Helper.init();
    h5Helper.readParticleData();
  } catch (const std::exception& e)
  {
    std::cerr<<e.what()<<std::endl;
    return EXIT_FAILURE;
  }
  // 3. Copy data to GPU
  particles_1.copyToGPU();
  Particles particles_2(particles_1);
  particles_2.copyToGPU();
  std::vector<Particles*> particles{ 
    &particles_1, 
    &particles_2
  };

  // Start the time
  auto startTime = std::chrono::high_resolution_clock::now();



  // 4. Run the loop - calculate new Particle positions.
  for (int s = 0; s < steps; s++)
  {
    calculate_velocity(*particles[s % 2], *particles[(s + 1) % 2], N, dt);
    /// In step 4 - fill in the code to store Particle snapshots.
    if (writeFreq > 0 && (s % writeFreq == 0))
    {
      
      /*float4 comOnGPU = {0.0f, 0.0f, 0.0f, 0.f};
      
      h5Helper.writeParticleData(s / writeFreq);
      h5Helper.writeCom(comOnGPU.x, comOnGPU.y, comOnGPU.z, comOnGPU.w, s / writeFreq);*/
    }
  }// for s ...
  // 5. In steps 3 and 4 -  Compute center of gravity
  float4 comOnGPU = {0.0f, 0.0f, 0.0f, 0.f};
  std::vector<float4> helper(N);
  #pragma acc enter data create(helper.data()[N])
  centerOfMassGPU(*particles[steps & 1ul], helper.data(), N);
  #pragma acc update host(helper.data()[N])
  comOnGPU.x = helper[0].x;
  comOnGPU.y = helper[0].y;
  comOnGPU.z = helper[0].z;
  comOnGPU.w = helper[0].w;

  // Stop watchclock
  const auto   endTime = std::chrono::high_resolution_clock::now();
  const double time    = (endTime - startTime) / std::chrono::milliseconds(1);
  printf("Time: %f s\n", time / 1000);

  // 5. Copy data from GPU back to CPU.
  particles[steps % 2]->copyToHost();
  if (steps % 2 > 0) {
    std::memcpy(particles[0]->positions, particles[1]->positions, sizeof(float4) * N);
    std::memcpy(particles[0]->velocities, particles[1]->velocities, sizeof(float3) * N);
  }

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  /// Calculate center of gravity
  float4 comOnCPU = centerOfMassCPU(md);


  // std::cout<<"Center of mass on CPU:"<<std::endl
  //   << comOnCPU.x <<", "
  //   << comOnCPU.y <<", "
  //   << comOnCPU.z <<", "
  //   << comOnCPU.w
  //   << std::endl;

  // std::cout<<"Center of mass on GPU:"<<std::endl
  //   << comOnGPU.x <<", "
  //   << comOnGPU.y <<", "
  //   << comOnGPU.z <<", "
  //   << comOnGPU.w
  //   <<std::endl;

  // Store final positions of the particles into a file
  h5Helper.writeComFinal(comOnGPU.x, comOnGPU.y, comOnGPU.z, comOnGPU.w);
  h5Helper.writeParticleDataFinal();
 #pragma acc exit data delete(helper.data()[0:N])
  return EXIT_SUCCESS;
}// end of main
//----------------------------------------------------------------------------------------------------------------------

