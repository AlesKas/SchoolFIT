/*
 * Architektura procesoru (ACH 2016)
 * Projekt c. 1 (nbody)
 * Login: xlogin00
 */

#include "nbody.h"

void particles_simulate(t_particles &p)
{
    int i, j, k;

    t_velocities velocities = {};

    for (k = 0; k < STEPS; k++)
    {
            //vynulovani mezisouctu
        for (i = 0; i < N; i++)
        {
            velocities[i].x = 0.0f;
            velocities[i].y = 0.0f;
            velocities[i].z = 0.0f;
        }
            //vypocet nove rychlosti
        for (i = 0; i < N; i++)
        {
            for (j = 0; j < N; j++)
            {
                calculate_gravitation_velocity(p[j], p[i], velocities[j]);
                calculate_collision_velocity(p[j], p[i], velocities[j]);
            }
        }
            //ulozeni rychlosti a posun castic
        for (i = 0; i < N; i++)
        {
            p[i].vel_x += velocities[i].x;
            p[i].vel_y += velocities[i].y;
            p[i].vel_z += velocities[i].z;

            p[i].pos_x += p[i].vel_x * DT;
            p[i].pos_y += p[i].vel_y * DT;
            p[i].pos_z += p[i].vel_z * DT;
        }
    }
}


void particles_read(FILE *fp, t_particles &p)
{
    for (int i = 0; i < N; i++)
    {
        fscanf(fp, "%f %f %f %f %f %f %f \n",
            &p[i].pos_x, &p[i].pos_y, &p[i].pos_z,
            &p[i].vel_x, &p[i].vel_y, &p[i].vel_z,
            &p[i].weight);
    }
}

void particles_write(FILE *fp, t_particles &p)
{
    for (int i = 0; i < N; i++)
    {
        fprintf(fp, "%10.10f %10.10f %10.10f %10.10f %10.10f %10.10f %10.10f \n",
            p[i].pos_x, p[i].pos_y, p[i].pos_z,
            p[i].vel_x, p[i].vel_y, p[i].vel_z,
            p[i].weight);
    }
}
