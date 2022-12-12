/**
 * @file      nbody.cpp
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

#include <math.h>
#include <cfloat>
#include "nbody.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                       Declare following structs / classes                                          //
//                                  If necessary, add your own classes / routines                                     //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void calculate_velocity(const Particles& p_in,
                                    Particles& p_out,
                                    const int        N,
                                    const float      dt)
{ 
  #pragma acc parallel loop vector_length(1024) present(p_in, p_out)
  for (int i = 0; i < N; i++) {
    float3 acc;
    float4 particle = float4(p_in.positions[i].x, p_in.positions[i].y, p_in.positions[i].z, p_in.positions[i].w);
    float3 particle_vel = float3(p_in.velocities[i].x, p_in.velocities[i].y, p_in.velocities[i].z);

    #pragma acc loop seq
    for(int particle2idx = 0; particle2idx < N; particle2idx++) {
      float4 particle2 = float4(p_in.positions[particle2idx].x, p_in.positions[particle2idx].y, p_in.positions[particle2idx].z, p_in.positions[particle2idx].w);
      float3 particle2_vel = float3(p_in.velocities[particle2idx].x, p_in.velocities[particle2idx].y, p_in.velocities[particle2idx].z);
      float3 diff = float3(particle.x - particle2.x, particle.y - particle2.y, particle.z - particle2.z);
      float r = sqrtf((diff.x * diff.x) + (diff.y * diff.y) + (diff.z * diff.z));

      float F = -G * particle.w * particle2.w / (r * r + FLT_MIN);
      if (r > COLLISION_DISTANCE) {
        float r3 = r * r * r + FLT_MIN;
        float G_dt_r3 = -G * dt / r3;
        float Fg_dt_m2_r = G_dt_r3 * particle2.w;
        acc.x += Fg_dt_m2_r * diff.x;
        acc.y += Fg_dt_m2_r * diff.y;
        acc.z += Fg_dt_m2_r * diff.z;
      } else {
        float weightSum = particle.w + particle2.w;
        float weightDiff = particle.w - particle2.w;
        acc.x += (r > 0.0f) ? (((particle_vel.x * weightDiff) + (2 * particle2.w * particle2.x)) / weightSum) - particle_vel.x : 0.0f;
        acc.y += (r > 0.0f) ? (((particle_vel.y * weightDiff) + (2 * particle2.w * particle2.y)) / weightSum) - particle_vel.y : 0.0f;
        acc.z += (r > 0.0f) ? (((particle_vel.z * weightDiff) + (2 * particle2.w * particle2.z)) / weightSum) - particle_vel.z : 0.0f;
      }
    }
    float3 velocity = float3(particle_vel.x + acc.x, particle_vel.y + acc.y, particle_vel.z + acc.z);
    p_out.velocities[i].x = velocity.x;
    p_out.velocities[i].y = velocity.y;
    p_out.velocities[i].z = velocity.z;

    p_out.positions[i].x = particle.x + (velocity.x * dt);
    p_out.positions[i].y = particle.y + (velocity.y * dt);
    p_out.positions[i].z = particle.z + (velocity.z * dt);
  }
}

/// Compute center of gravity
float4 centerOfMassGPU(const Particles& p_in,
                       const int        N)
{

  return {0.0f, 0.0f, 0.0f, 0.0f};
}// end of centerOfMassGPU
//----------------------------------------------------------------------------------------------------------------------

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// Compute center of mass on CPU
float4 centerOfMassCPU(MemDesc& memDesc)
{
  float4 com = {0 ,0, 0, 0};

  for(int i = 0; i < memDesc.getDataSize(); i++)
  {
    // Calculate the vector on the line connecting points and most recent position of center-of-mass
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
}// end of centerOfMassCPU
//----------------------------------------------------------------------------------------------------------------------
