/*
 * Architektura procesoru (ACH 2016)
 * Projekt c. 1 (nbody)
 * Login: xlogin00
 */

#ifndef __NBODY_H__
#define __NBODY_H__

#include <cstdlib>
#include <cstdio>
#include "velocity.h"

using t_particles   = t_particle[N];
using t_velocities  = t_velocity[N];

void particles_simulate(t_particles &p);

void particles_read(FILE *fp, t_particles &p);

void particles_write(FILE *fp, t_particles &p);

#endif /* __NBODY_H__ */
