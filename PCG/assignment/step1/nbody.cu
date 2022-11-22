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

__global__ void calculate_velocity(t_particles p_in, t_particles p_out, int N, float dt) {
  unsigned int threadId = blockDim.x * blockIdx.x + threadIdx.x;
  for (int i = threadId; i < N; i += blockDim.x * gridDim.x) {
    float r, dx, dy, dz;
    float vx, vy, vz;
    float accX = 0;
    float accY = 0;
    float accZ = 0;

    float pos_x = p_in.pos_x[i];
    float pos_y = p_in.pos_y[i];
    float pos_z = p_in.pos_z[i];
    float vel_x = p_in.vel_x[i];
    float vel_y = p_in.vel_y[i];
    float vel_z = p_in.vel_z[i];
    float weight = p_in.weight[i];
    for (int particle = 0; particle < N; particle++) {
      dx = pos_x - p_in.pos_x[particle];
      dy = pos_y - p_in.pos_y[particle];
      dz = pos_z - p_in.pos_z[particle];
      float p2weight = p_in.weight[particle];
      r = sqrt(dx*dx + dy*dy + dz*dz);
      if (r > COLLISION_DISTANCE) {
        // Větev pro gravitaci, používám vzorce z velocity.cpp
        float r3, G_dt_r3, Fg_dt_m2_r;
        r3 = r * r * r + FLT_MIN;
        // Fg*dt/m1/r = G*m1*m2*dt / r^3 / m1 = G*dt/r^3 * m2
        G_dt_r3 = -G * dt / r3;
        Fg_dt_m2_r = G_dt_r3 * p2weight;
        // vx = - Fx*dt/m2 = - Fg*dt/m2 * dx/r = - Fg*dt/m2/r * dx
        vx = Fg_dt_m2_r * dx;
        vy = Fg_dt_m2_r * dy;
        vz = Fg_dt_m2_r * dz;

        accX += vx;
        accY += vy;
        accZ += vz;
      } else {
        // Větev pro kolize
        float p2vel_x = p_in.vel_x[particle];
        float p2vel_y = p_in.vel_y[particle];
        float p2vel_z = p_in.vel_z[particle];

        r = sqrt(dx*dx + dy*dy + dz*dz);
        float weightSum = weight + p_in.weight[particle];

        // weight * vel_x - p2weight * vel_x = vel_x * (weight - p2weight)
        float weightDiff = weight - p2weight;
        vx = ((vel_x * weightDiff + 2 * p2weight * p2vel_x) / weightSum) - vel_x;
        vy = ((vel_y * weightDiff + 2 * p2weight * p2vel_y) / weightSum) - vel_y;
        vz = ((vel_z * weightDiff + 2 * p2weight * p2vel_z) / weightSum) - vel_z;

        accX += (r > 0.0f) ? vx : 0.0f;
        accY += (r > 0.0f) ? vy : 0.0f;
        accZ += (r > 0.0f) ? vz : 0.0f;
      }
    }
    p_out.vel_x[i] = vel_x + accX;
    p_out.vel_y[i] = vel_y + accY;
    p_out.vel_z[i] = vel_z + accZ;

    p_out.pos_x[i] = pos_x + p_out.vel_x[i] * dt;
    p_out.pos_y[i] = pos_y + p_out.vel_y[i] * dt;
    p_out.pos_z[i] = pos_z + p_out.vel_z[i] * dt;
  }
}

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
