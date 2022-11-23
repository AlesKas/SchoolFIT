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
    extern __shared__ float sharedMem[];
    // Sdílená paměť bude mít podobu
    /*[
        0 až blockDim.x - 1 : pos_x
        blockDim.x až blockDim.x * 2 - 1 : pos_y
        blockDim.x * 2 až blockDim.x * 3 - 1 : pos_z
        blockDim.x * 3 až blockDim.x * 4 - 1 : vel_x
        blockDim.x * 4 až blockDim.x * 5 - 1 : vel_y
        blockDim.x * 5 až blockDim.x * 6 - 1 : vel_z
        blockDim.x * 6 až blockDim.x * 7 - 1 : weight
    ]*/
    // Ukazatele na příslušné části paměti
    float* pos_x_SM = sharedMem;
    float* pos_y_SM = &sharedMem[blockDim.x];
    float* pos_z_SM = &sharedMem[blockDim.x * 2];
    float* vel_x_SM = &sharedMem[blockDim.x * 3];
    float* vel_y_SM = &sharedMem[blockDim.x * 4];
    float* vel_z_SM = &sharedMem[blockDim.x * 5];
    float* weight_SM = &sharedMem[blockDim.x * 6];

    // Budeme se posouvat pouze o dlaždice, aby nám vystačila sdílená paměť
    int tileWidth = blockDim.x;
    int numberOfTiles = ceil(float(N) / tileWidth);
    // Pokud by nám nestačila vlákna
    int numberOfThreads = gridDim.x * blockDim.x;
    int numberOfGridsNeeded = ceil(float(N) / numberOfThreads);
    for (int grid = 0; grid < numberOfGridsNeeded; grid++) {
        // Výpočet globálního threadId
        int threadId = (grid * numberOfThreads) + (blockDim.x * blockIdx.x + threadIdx.x);
        float r, dx, dy, dz;
        float vx, vy, vz;
        float accX = 0;
        float accY = 0;
        float accZ = 0;

        float pos_x = p_in.pos_x[threadId];
        float pos_y = p_in.pos_y[threadId];
        float pos_z = p_in.pos_z[threadId];
        float vel_x = p_in.vel_x[threadId];
        float vel_y = p_in.vel_y[threadId];
        float vel_z = p_in.vel_z[threadId];
        float weight = p_in.weight[threadId];

        for (int tile = 0; tile < numberOfTiles; tile++) {
            int tileNumber = tile * blockDim.x;
            int threadNumber = tileNumber + threadIdx.x;
            pos_x_SM[threadIdx.x] = p_in.pos_x[threadNumber];
            pos_y_SM[threadIdx.x] = p_in.pos_y[threadNumber];
            pos_z_SM[threadIdx.x] = p_in.pos_z[threadNumber];
            vel_x_SM[threadIdx.x] = p_in.vel_x[threadNumber];
            vel_y_SM[threadIdx.x] = p_in.vel_y[threadNumber];
            vel_z_SM[threadIdx.x] = p_in.vel_z[threadNumber];
            weight_SM[threadIdx.x] = p_in.weight[threadNumber];
            __syncthreads();
            for (int particle = 0; particle < tileWidth; particle++) {
                dx = pos_x - pos_x_SM[particle];
                dy = pos_y - pos_y_SM[particle];
                dz = pos_z - pos_z_SM[particle];
                float p2weight = weight_SM[particle];
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
                    float p2vel_x = pos_x_SM[particle];
                    float p2vel_y = pos_y_SM[particle];
                    float p2vel_z = pos_z_SM[particle];
                    float weightSum = weight + p2weight;
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
            __syncthreads();
        }
        if (threadId < N) {
            p_out.vel_x[threadId] = vel_x + accX;
            p_out.vel_y[threadId] = vel_y + accY;
            p_out.vel_z[threadId] = vel_z + accZ;

            p_out.pos_x[threadId] = pos_x + p_out.vel_x[threadId] * dt;
            p_out.pos_y[threadId] = pos_y + p_out.vel_y[threadId] * dt;
            p_out.pos_z[threadId] = pos_z + p_out.vel_z[threadId] * dt;
        }
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
    int warpsCount = blockDim.x / 32;
    // Sdílená paměť bude mít podobu
    /*[
        0 až warpsCount - 1 : X
        warpsCount až warpsCount * 2 - 1 : Y
        warpsCount * 2 až warpsCount * 3 - 1 : Z
        warpsCount * 3 až warpsCount * 4 - 1 : weight
    ]*/
    extern __shared__ volatile float sharedMemCOM[];
    volatile float *x_SM = sharedMemCOM;
    volatile float *y_SM = &sharedMemCOM[warpsCount];
    volatile float *z_SM = &sharedMemCOM[warpsCount * 2];
    volatile float *weight_SM = &sharedMemCOM[warpsCount * 3];

    float4 acc = {0.0f, 0.0f, 0.0f, 0.0f};

    // Pokud by nám nestačila vlákna
    int numberOfThreads = gridDim.x * blockDim.x;
    int numberOfGridsNeeded = ceil(float(N) / numberOfThreads);
    for (int grid = 0; grid < numberOfGridsNeeded; grid++) {
        // Výpočet globálního threadId
        int threadId = (grid * numberOfThreads) + (blockDim.x * blockIdx.x + threadIdx.x);
        float pos_x = p.pos_x[threadId];
        float pos_y = p.pos_y[threadId];
        float pos_z = p.pos_z[threadId];
        float weight = p.weight[threadId];
        float dw = (weight > 0.0f) ? 1.0f : 0.0f;

        for (int stride = 16; stride > 0; stride >>= 1) {
            float p2weight = __shfl_down_sync(0xffffffff, weight, stride);
            weight += p2weight;
            dw = (weight > 0.0f) ? (p2weight / weight) :0.0f;
            pos_x += (__shfl_down_sync(0xffffffff, pos_x, stride) - pos_x) * dw;
            pos_y += (__shfl_down_sync(0xffffffff, pos_y, stride) - pos_y) * dw;
            pos_z += (__shfl_down_sync(0xffffffff, pos_z, stride) - pos_z) * dw;
        }

        if ((threadIdx.x % 32 == 0) && threadId < N) {
            dw = ((weight + acc.w) > 0.0f) ? (weight / (weight + acc.w)) : 0.0f;
            acc.x += (pos_x - acc.x) * dw;
            acc.y += (pos_y - acc.y) * dw;
            acc.z += (pos_z - acc.z) * dw;
            acc.w += weight;
        }
    }
    unsigned int warpNum = threadIdx.x / 32;
    if (threadIdx.x % 32u == 0) {
        x_SM[warpNum] = acc.x;
        y_SM[warpNum] = acc.y;
        z_SM[warpNum] = acc.z;
        weight_SM[warpNum] = acc.w;
        // printf("%d %d\n", warpsCount >> 1ul, 1);
    }
    __syncthreads();

    for (int stride = warpsCount >> 1ul; stride > 16; stride >>= 1ul) {
        if (threadIdx.x < stride) {
            float weight = weight_SM[threadIdx.x + stride];
            float dw = ((weight + weight_SM[threadIdx.x]) > 0.0f) ? (weight / (weight + weight_SM[threadIdx.x])): 0.0f;

            x_SM[threadIdx.x] += (x_SM[threadIdx.x + stride] - x_SM[threadIdx.x]) * dw;
            y_SM[threadIdx.x] += (y_SM[threadIdx.x + stride] - y_SM[threadIdx.x]) * dw;
            z_SM[threadIdx.x] += (z_SM[threadIdx.x + stride] - z_SM[threadIdx.x]) * dw;
            weight_SM[threadIdx.x] += weight_SM[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x < min(32, warpsCount)) {
        float pos_x = x_SM[threadIdx.x];
        float pos_y = y_SM[threadIdx.x];
        float pos_z = z_SM[threadIdx.x];
        float weight = weight_SM[threadIdx.x];
        for (int stride = min(16, warpsCount >> 1ul); stride > 0; stride >>= 1u) {
            float w2 = __shfl_down_sync(0xffffffff, weight, stride);
            weight += w2;
            float dw = ((weight) > 0.0f) ? (w2 / weight) : 0.0f;
            pos_x += (__shfl_down_sync(0xffffffff, pos_x, stride) - pos_x) * dw;
            pos_y += (__shfl_down_sync(0xffffffff, pos_y, stride) - pos_y) * dw;
            pos_z += (__shfl_down_sync(0xffffffff, pos_z, stride) - pos_z) * dw;
        }
        if (threadIdx.x == 0) {
            while (atomicExch(lock, 1) != 0);
            float dw = ((weight + *comW) > 0.0f) ? (weight / (weight + *comW)) : 0.0f;
            *comX += (pos_x - *comX) * dw;
            *comY += (pos_y - *comY) * dw;
            *comZ += (pos_z - *comZ) * dw;
            *comW += weight;
            atomicExch(lock, 0);
        }
    }
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
