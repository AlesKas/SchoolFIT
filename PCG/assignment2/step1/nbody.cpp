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

/// Compute gravitation velocity
void calculate_gravitation_velocity(const Particles& p,
                                    Velocities&      tmp_vel,
                                    const int        N,
                                    const float      dt)
{
  #pragma acc parallel loop vector_length(1024) present(p, tmp_vel)
  for (int i = 0; i < N; i++) {
    float3 acc;
    float4 particle = float4(p.positions[i].x, p.positions[i].y, p.positions[i].z, p.positions[i].w);

    #pragma acc loop seq
    for(int particle2idx = 0; particle2idx < N; particle2idx++) {
      float4 particle2 = float4(p.positions[particle2idx].x, p.positions[particle2idx].y, p.positions[particle2idx].z, p.positions[particle2idx].w);
      float3 diff = float3(particle.x - particle2.x, particle.y - particle2.y, particle.z - particle2.z);
      float r = sqrtf((diff.x * diff.x) + (diff.y * diff.y) + (diff.z * diff.z));

      float F = -G * particle.w * particle2.w / (r * r + FLT_MIN);
      float r3 = r * r * r + FLT_MIN;
      float G_dt_r3 = -G * dt / r3;
      float Fg_dt_m2_r = G_dt_r3 * particle2.w;
      acc.x += Fg_dt_m2_r * diff.x;
      acc.y += Fg_dt_m2_r * diff.y;
      acc.z += Fg_dt_m2_r * diff.z;
    }
    tmp_vel.velocities[i].x += acc.x;
    tmp_vel.velocities[i].y += acc.y;
    tmp_vel.velocities[i].z += acc.z;
  }
}// end of calculate_gravitation_velocity
//----------------------------------------------------------------------------------------------------------------------

void calculate_collision_velocity(const Particles& p,
                                  Velocities&      tmp_vel,
                                  const int        N,
                                  const float      dt)
{
  #pragma acc parallel loop vector_length(1024) present(p, tmp_vel)
  for(int i = 0; i < N; i++) {
    float3 acc;
    float4 particle = float4(p.positions[i].x, p.positions[i].y, p.positions[i].z, p.positions[i].w);
    float3 particle_vel = float3(p.velocities[i].x, p.velocities[i].y, p.velocities[i].z);

    #pragma acc loop seq
    for (int particle2idx = 0; particle2idx < N; particle2idx++) {
      float4 particle2 = float4(p.positions[particle2idx].x, p.positions[particle2idx].y, p.positions[particle2idx].z, p.positions[particle2idx].w);
      float3 particle2_vel = float3(p.velocities[particle2idx].x, p.velocities[particle2idx].y, p.velocities[particle2idx].z);
      float3 diff = float3(particle.x - particle2.x, particle.y - particle2.y, particle.z - particle2.z);

      float r = sqrtf((diff.x * diff.x) + (diff.y * diff.y) + (diff.z * diff.z));
      float weightSum = particle.w + particle2.w;
      float weightDiff = particle.w - particle2.w;

      if (r > 0.0f && r < COLLISION_DISTANCE) {
        acc.x += (((particle_vel.x * weightDiff) + (2 * particle2.w * particle2.x)) / weightSum) - particle_vel.x;
        acc.y += (((particle_vel.y * weightDiff) + (2 * particle2.w * particle2.y)) / weightSum) - particle_vel.y;
        acc.z += (((particle_vel.z * weightDiff) + (2 * particle2.w * particle2.z)) / weightSum) - particle_vel.z;
      }
    }
    tmp_vel.velocities[i].x += acc.x;
    tmp_vel.velocities[i].y += acc.y;
    tmp_vel.velocities[i].z += acc.z;
  }
}// end of calculate_collision_velocity
//----------------------------------------------------------------------------------------------------------------------

/// Update particle position
void update_particle(const Particles& p,
                     Velocities&      tmp_vel,
                     const int        N,
                     const float      dt)
{
  #pragma acc parallel loop vector_length(1024) present(p, tmp_vel)
  for(int i = 0; i < N; i++) {
    float3 velocity = float3(p.velocities[i].x + tmp_vel.velocities[i].x, p.velocities[i].y + tmp_vel.velocities[i].y, p.velocities[i].z + tmp_vel.velocities[i].z);
    p.velocities[i].x = velocity.x;
    p.velocities[i].y = velocity.y;
    p.velocities[i].z = velocity.z;

    p.positions[i].x += velocity.x * dt;
    p.positions[i].y += velocity.y * dt;
    p.positions[i].z += velocity.z * dt;
  }
}// end of update_particle
//----------------------------------------------------------------------------------------------------------------------



/// Compute center of gravity
float4 centerOfMassGPU(const Particles& p,
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
