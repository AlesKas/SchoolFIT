/**
 * @file      nbody.ch
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
 * @date      15 November  2022, 14:10 (revised) \n
 *
 */

#ifndef __NBODY_H__
#define __NBODY_H__

#include <iostream>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include  <cmath>
#include "h5Helper.h"

/// Gravity constant
constexpr float G = 6.67384e-11f;

/// Collision distance threshold
constexpr float COLLISION_DISTANCE = 0.01f;

/**
 * @struct float4
 * Structure that mimics CUDA float4
 */
struct float3 {
  float x;
  float y;
  float z;
  float3() : x(0.0f), y(0.0f), z(0.0f) {}
  float3(float val) : x(val), y(val), z(val) {}
  float3(float x, float y, float z) : x(x), y(y), z(z) {}
};

struct float4
{
  float x;
  float y;
  float z;
  float w;
  
  float4() : x(0.0f), y(0.0f), z(0.0f), w(0.0f) {}
  float4(float val) : x(val), y(val), z(val), w(val) {}
  float4(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {}
};

/// Define sqrtf from CUDA libm library
#pragma acc routine(sqrtf) seq

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                       Declare following structs / classes                                          //
//                                  If necessary, add your own classes / routines                                     //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * Structure with particle data
 */
struct Particles
{
  // Fill the structure holding the particle/s data
  // It is recommended to implement constructor / destructor and copyToGPU and copyToCPU routines
  float4* positions;
  float3* velocities;
  const int size = 0;
  Particles(int count) : size(count) {
    positions = new float4[count];
    velocities = new float3[count];
    #pragma acc enter data copyin(this)
    #pragma acc enter data create(positions[size])
    #pragma acc enter data create(velocities[size])
  }
  ~Particles() {
    #pragma acc exit data delete(positions)
    #pragma acc exit data delete(velocities)
    #pragma acc exit data delete(this)
    delete[] positions;
    delete[] velocities;
  }
  void copyToGPU() {
    #pragma acc update device(positions[size])
    #pragma acc update device(velocities[size])
  }
  void copyToHost() {
    #pragma acc update host(positions[size])
    #pragma acc update host(velocities[size])
  }
};// end of Particles
//----------------------------------------------------------------------------------------------------------------------

/**
 * @struct Velocities
 * Velocities of the particles
 */
struct Velocities
{
  // Fill the structure holding the particle/s data
  // It is recommended to implement constructor / destructor and copyToGPU and copyToCPU routines
  const int size = 0;
  float3* velocities;
  Velocities(int count) : size(count) {
    velocities = new float3[count];
    #pragma acc enter data copyin(this)
    #pragma acc enter data create(velocities[size])
  }
  ~Velocities() {
    #pragma acc exit data delete(velocities)
    #pragma acc exit data delete(this)
    delete[] velocities;
  }
  void copyToGPU() {
    #pragma acc update device(velocities[size])
  }
  void copyToHost() {
    #pragma acc update host(velocities[size])
  }
  void Memset(int val) {
    std::memset(velocities, val, size * sizeof(float3));
    copyToGPU();
  }
  void MemsetOnDevice(int val) {
    #pragma acc parallel loop gang vector_length(32) present(this, velocities[size])
    for (int i = 0; i < size; i++) {
      velocities[i].x = float(val);
      velocities[i].y = float(val);
      velocities[i].z = float(val);
    }
  }
};// end of Velocities
//----------------------------------------------------------------------------------------------------------------------

/**
 * Compute gravitation velocity
 * @param [in]  p        - Particles
 * @param [out] tmp_vel  - Temporal velocity
 * @param [in ] N        - Number of particles
 * @param [in]  dt       - Time step size
 */
void calculate_gravitation_velocity(const Particles& p,
                                    Velocities&      tmp_vel,
                                    const int        N,
                                    const float      dt);

/**
 * Calculate collision velocity
 * @param [in]  p        - Particles
 * @param [out] tmp_vel  - Temporal velocity
 * @param [in ] N        - Number of particles
 * @param [in]  dt       - Time step size
 */
void calculate_collision_velocity(const Particles& p,
                                  Velocities&      tmp_vel,
                                  const int        N,
                                  const float      dt);

/**
 * Update particle position
 * @param [in]  p        - Particles
 * @param [out] tmp_vel  - Temporal velocity
 * @param [in ] N        - Number of particles
 * @param [in]  dt       - Time step size
 */
void update_particle(const Particles& p,
                     Velocities&      tmp_vel,
                     const int        N,
                     const float      dt);



/**
 * Compute center of gravity - implement in steps 3 and 4.
 * @param [in] p - Particles
 * @param [in] N - Number of particles
 * @return Center of Mass [x, y, z] and total weight[w]
 */
float4 centerOfMassGPU(const Particles& p,
                       const int        N);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * Compute center of mass on CPU
 * @param memDesc
 * @return centre of gravity
 */
float4 centerOfMassCPU(MemDesc& memDesc);

#endif /* __NBODY_H__ */
