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
    extern __shared__ float cache[];
    float *posX = cache;
    float *posY = &cache[blockDim.x];
    float *posZ = &cache[blockDim.x * 2];
    float *velX = &cache[blockDim.x * 3];
    float *velY = &cache[blockDim.x * 4];
    float *velZ = &cache[blockDim.x * 5];
    float *weights = &cache[blockDim.x * 6];

    unsigned int threadsTotal = gridDim.x * blockDim.x;
    unsigned int gridSteps = ceil(float(N) / threadsTotal);
    unsigned int tileWidth = blockDim.x;
    unsigned int tileCount = ceil(float(N) / tileWidth);

    /// Grid stride loop (if there's not enough total threads to cover all particles)
    for (unsigned int gridIdx = 0; gridIdx < gridSteps; gridIdx++) {
        float dx, dy, dz;
        float accVelocityX = 0;
        float accVelocityY = 0;
        float accVelocityZ = 0;

        unsigned int globalIdx = (gridIdx * threadsTotal) + (blockIdx.x * blockDim.x + threadIdx.x);

        bool inBounds = globalIdx < N;
        float p1_x = inBounds ? p_in.pos_x[globalIdx] : 0.0f;
        float p1_y = inBounds ? p_in.pos_y[globalIdx] : 0.0f;
        float p1_z = inBounds ? p_in.pos_z[globalIdx] : 0.0f;
        float p1_vel_x = inBounds ? p_in.vel_x[globalIdx] : 0.0f;
        float p1_vel_y = inBounds ? p_in.vel_y[globalIdx] : 0.0f;
        float p1_vel_z = inBounds ? p_in.vel_z[globalIdx] : 0.0f;
        float p1_weight = inBounds ? p_in.weight[globalIdx] : 0.0f;

        /// Loop over all tiles with each thread
        for (unsigned int tileIdx = 0; tileIdx < tileCount; tileIdx++) {
            unsigned int tileOffset = tileIdx * blockDim.x;
            unsigned int threadOffset = tileOffset + threadIdx.x;
            posX[threadIdx.x] = (threadOffset < N) ? p_in.pos_x[threadOffset] : 0.0f;
            posY[threadIdx.x] = (threadOffset < N) ? p_in.pos_y[threadOffset] : 0.0f;
            posZ[threadIdx.x] = (threadOffset < N) ? p_in.pos_z[threadOffset] : 0.0f;
            velX[threadIdx.x] = (threadOffset < N) ? p_in.vel_x[threadOffset] : 0.0f;
            velY[threadIdx.x] = (threadOffset < N) ? p_in.vel_y[threadOffset] : 0.0f;
            velZ[threadIdx.x] = (threadOffset < N) ? p_in.vel_z[threadOffset] : 0.0f;
            weights[threadIdx.x] = (threadOffset < N) ? p_in.weight[threadOffset] : 0.0f;

            /// Synchronize threads before using shared memory
            __syncthreads();

            /// Loop over all points in a single tile
            for (int p2_idx = 0; p2_idx < tileWidth; p2_idx++) {
                dx = p1_x - posX[p2_idx];
                dy = p1_y - posY[p2_idx];
                dz = p1_z - posZ[p2_idx];
                float rr = dx * dx + dy * dy + dz * dz;
                float r = sqrt(rr);

                if (r > COLLISION_DISTANCE) {
                    // Fg*dt/m1/r = G*m1*m2*dt / r^3 / m1 = G*dt/r^3 * m2
                    // vx = - Fx*dt/m2 = - Fg*dt/m2 * dx/r = - Fg*dt/m2/r * dx
                    float r3 = rr * r + FLT_MIN;
                    float G_dt_r3 = -G * dt / r3;
                    float Fg_dt_m2_r = G_dt_r3 * weights[p2_idx];
                    accVelocityX += Fg_dt_m2_r * dx;
                    accVelocityY += Fg_dt_m2_r * dy;
                    accVelocityZ += Fg_dt_m2_r * dz;
                } else {
                    float weightSum = p1_weight + weights[p2_idx];
                    float weightDiff = p1_weight - weights[p2_idx];
                    float p2_w2 = 2 * weights[p2_idx];

                    bool colliding = r > 0.0f;
                    accVelocityX += colliding ? ((p1_vel_x * weightDiff + p2_w2 * velX[p2_idx]) / weightSum) - p1_vel_x
                                              : 0.0f;
                    accVelocityY += colliding ? ((p1_vel_y * weightDiff + p2_w2 * velY[p2_idx]) / weightSum) - p1_vel_y
                                              : 0.0f;
                    accVelocityZ += colliding ? ((p1_vel_z * weightDiff + p2_w2 * velZ[p2_idx]) / weightSum) - p1_vel_z
                                              : 0.0f;
                }
            }

            /// Wait for all threads to finish to avoid overwritten shared memory
            __syncthreads();
        }

        if (globalIdx < N) {
            p_out.vel_x[globalIdx] = p1_vel_x + accVelocityX;
            p_out.vel_y[globalIdx] = p1_vel_y + accVelocityY;
            p_out.vel_z[globalIdx] = p1_vel_z + accVelocityZ;

            p_out.pos_x[globalIdx] = p1_x + p_out.vel_x[globalIdx] * dt;
            p_out.pos_y[globalIdx] = p1_y + p_out.vel_y[globalIdx] * dt;
            p_out.pos_z[globalIdx] = p1_z + p_out.vel_z[globalIdx] * dt;
        }
    }
}// end of calculate_gravitation_velocity

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
