/**
 * @File nbody.h
 *
 * Header file of your implementation to the N-Body problem
 *
 * Paralelní programování na GPU (PCG 2022)
 * Projekt c. 1 (cuda)
 * Login: xlogin00
 */

#ifndef __NBODY_H__
#define __NBODY_H__

#include <cstdlib>
#include <cstdio>
#include "h5Helper.h"

/* Gravitation constant */
constexpr float G =  6.67384e-11f;
constexpr float COLLISION_DISTANCE = 0.01f;


/**
 * Particles data structure
 */
typedef struct
{
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//                 FILL IN: Particle data structure optimal for the use on GPU (step 0)                             //
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
} t_particles;

/**
/* Velocities data structure (to be used as buffer for partial results)
 */
typedef struct
{
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//                 FILL IN: Velocities data structure optimal for the use on GPU (step 0)                           //
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
} t_velocities;


/**
 * CUDA kernel to calculate gravitation velocity
 * @param p       - particles
 * @param tmp_vel - temp array for velocities
 * @param N       - Number of particles
 * @param dt      - Size of the time step
 */
__global__ void calculate_gravitation_velocity(t_particles  p,
                                               t_velocities tmp_vel,
                                               int          N,
                                               float        dt);

/**
 * CUDA kernel to calculate collision velocity
 * @param p       - particles
 * @param tmp_vel - temp array for velocities
 * @param N       - Number of particles
 * @param dt      - Size of the time step
 */
__global__ void calculate_collision_velocity(t_particles  p,
                                             t_velocities tmp_vel,
                                             int          N,
                                             float        dt);

/**
 * CUDA kernel to update particles
 * @param p       - particles
 * @param tmp_vel - temp array for velocities
 * @param N       - Number of particles
 * @param dt      - Size of the time step
 */
__global__ void update_particle(t_particles  p,
                                t_velocities tmp_vel,
                                int           N,
                                float         dt);

/**
 * CUDA kernel to update particles
 * @param p       - particles
 * @param comX    - pointer to a center of mass position in X
 * @param comY    - pointer to a center of mass position in Y
 * @param comZ    - pointer to a center of mass position in Z
 * @param comW    - pointer to a center of mass weight
 * @param lock    - pointer to a user-implemented lock
 * @param N       - Number of particles
 */
__global__ void centerOfMass(t_particles p,
                             float*      comX,
                             float*      comY,
                             float*      comZ,
                             float*      comW,
                             int*        lock,
                             const int   N);

/**
 * CPU implementation of the Center of Mass calculation
 * @param memDesc - Memory descriptor of particle data on CPU side
 */
float4 centerOfMassCPU(MemDesc& memDesc);

#endif /* __NBODY_H__ */
