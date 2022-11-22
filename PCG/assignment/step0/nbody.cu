/**
 * @File nbody.cu
 *
 * Implementation of the N-Body problem
 *
 * Paralelní programování na GPU (PCG 2022)
 * Projekt c. 1 (cuda)
 * Login: xkaspa48
 */

#include <cmath>
#include <cfloat>
#include "nbody.h"

/**
 * CUDA kernel to calculate gravitation velocity
 * @param p       - particles
 * @param tmp_vel - temp array for velocities
 * @param N       - Number of particles
 * @param dt      - Size of the time step
 */
// Výpočet rychlosti, kterou částice dostane působením všech ostatních částic
__global__ void calculate_gravitation_velocity(t_particles p, t_velocities tmp_vel, int N, float dt) {
  unsigned int threadId = blockDim.x * blockIdx.x + threadIdx.x;
  for (int i = threadId; i < N; i += blockDim.x * gridDim.x) {
    float r, dx, dy, dz;
    float vx, vy, vz;

    // Rozhodl jsem se vytáhnout adresování p.pos_x/y/z z vnitřního for cyklu
    // Přístup do registru bude rychlejší než adresace paměti
    float pos_x = p.pos_x[i];
    float pos_y = p.pos_y[i];
    float pos_z = p.pos_z[i];
    float weight = p.weight[i];

    for (int particle = 0; particle < N; particle++) {
      float p2weight = p.weight[particle];
      dx = pos_x - p.pos_x[particle];
      dy = pos_y - p.pos_y[particle];
      dz = pos_z - p.pos_z[particle];
      r = sqrt(dx*dx + dy*dy + dz*dz);
      float r3, G_dt_r3, Fg_dt_m2_r;

      r3 = r * r * r + FLT_MIN;

      // Fg*dt/m1/r = G*m1*m2*dt / r^3 / m1 = G*dt/r^3 * m2
      G_dt_r3 = -G * dt / r3;
      Fg_dt_m2_r = G_dt_r3 * p2weight;

      // vx = - Fx*dt/m2 = - Fg*dt/m2 * dx/r = - Fg*dt/m2/r * dx
      vx = Fg_dt_m2_r * dx;
      vy = Fg_dt_m2_r * dy;
      vz = Fg_dt_m2_r * dz;

      tmp_vel.x[i] += (r > COLLISION_DISTANCE) ? vx : 0.0f;
      tmp_vel.y[i] += (r > COLLISION_DISTANCE) ? vy : 0.0f;
      tmp_vel.z[i] += (r > COLLISION_DISTANCE) ? vz : 0.0f;
    }
  }
}// end of calculate_gravitation_velocity
//----------------------------------------------------------------------------------------------------------------------

/**
 * CUDA kernel to calculate collision velocity
 * @param p       - particles
 * @param tmp_vel - temp array for velocities
 * @param N       - Number of particles
 * @param dt      - Size of the time step
 */
__global__ void calculate_collision_velocity(t_particles p, t_velocities tmp_vel, int N, float dt) {
  unsigned int threadId = blockDim.x * blockIdx.x + threadIdx.x;
  for (int i = threadId; i < N; i += blockDim.x * gridDim.x) {
    float r, dx, dy, dz;
    float vx, vy, vz;

    // Podobná úvaha jako o funkci výše
    float pos_x = p.pos_x[i];
    float pos_y = p.pos_y[i];
    float pos_z = p.pos_z[i];
    float vel_x = p.vel_x[i];
    float vel_y = p.vel_y[i];
    float vel_z = p.vel_z[i];
    float weight = p.weight[i];

    for (int particle = 0; particle < N; particle++) {
      // Opět jsem se dovolil inspirovat se velocity.cpp
      dx = pos_x - p.pos_x[particle];
      dy = pos_y - p.pos_y[particle];
      dz = pos_z - p.pos_z[particle];
      float p2vel_x = p.vel_x[particle];
      float p2vel_y = p.vel_y[particle];
      float p2vel_z = p.vel_z[particle];
      float p2weight = p.weight[particle];

      r = sqrt(dx*dx + dy*dy + dz*dz);
      float weightSum = weight + p.weight[particle];

      // weight * vel_x - p2weight * vel_x = vel_x * (weight - p2weight)
      float weightDiff = weight - p2weight;
      vx = ((vel_x * weightDiff + 2 * p2weight * p2vel_x) / weightSum) - vel_x;
      vy = ((vel_y * weightDiff + 2 * p2weight * p2vel_y) / weightSum) - vel_y;
      vz = ((vel_z * weightDiff + 2 * p2weight * p2vel_z) / weightSum) - vel_z;

      tmp_vel.x[i] += (r > 0.0f && r < COLLISION_DISTANCE) ? vx : 0.0f;
      tmp_vel.y[i] += (r > 0.0f && r < COLLISION_DISTANCE) ? vy : 0.0f;
      tmp_vel.z[i] += (r > 0.0f && r < COLLISION_DISTANCE) ? vz : 0.0f;
    }
  }
}// end of calculate_collision_velocity
//----------------------------------------------------------------------------------------------------------------------

/**
 * CUDA kernel to update particles
 * @param p       - particles
 * @param tmp_vel - temp array for velocities
 * @param N       - Number of particles
 * @param dt      - Size of the time step
 */
__global__ void update_particle(t_particles p, t_velocities tmp_vel, int N, float dt) {
  unsigned int threadId = blockDim.x * blockIdx.x + threadIdx.x;
  for (int i = threadId; i < N; i += blockDim.x * gridDim.x) {
    p.vel_x[i] += tmp_vel.x[i];
    p.pos_x[i] += p.vel_x[i] * dt;
    
    p.vel_y[i] += tmp_vel.y[i];
    p.pos_y[i] += p.vel_y[i] * dt;
    
    p.vel_z[i] += tmp_vel.z[i];
    p.pos_z[i] += p.vel_z[i] * dt;
  }
}// end of update_particle
//----------------------------------------------------------------------------------------------------------------------

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
__global__ void centerOfMass(t_particles p, float* comX, float* comY, float* comZ, float* comW, int* lock, const int N) {

}// end of centerOfMass
//----------------------------------------------------------------------------------------------------------------------

/**
 * CPU implementation of the Center of Mass calculation
 * @param particles - All particles in the system
 * @param N         - Number of particles
 */
__host__ float4 centerOfMassCPU(MemDesc& memDesc) {
  float4 com = {0 ,0, 0, 0};

  for(int i = 0; i < memDesc.getDataSize(); i++)
  {
    // Calculate the vector on the line connecting current body and most recent position of center-of-mass
    const float dx = memDesc.getPosX(i) - com.x;
    const float dy = memDesc.getPosY(i) - com.y;
    const float dz = memDesc.getPosZ(i) - com.z;

    // Calculate weight ratio only if at least one particle isn't massless
    const float dw = ((memDesc.getWeight(i) + com.w) > 0.0f)
                          ? ( memDesc.getWeight(i) / (memDesc.getWeight(i) + com.w)) : 0.0f;

    // Update position and weight of the center-of-mass according to the weight ration and vector
    com.x += dx * dw;
    com.y += dy * dw;
    com.z += dz * dw;
    com.w += memDesc.getWeight(i);
  }
  return com;
}// enf of centerOfMassCPU
//----------------------------------------------------------------------------------------------------------------------
