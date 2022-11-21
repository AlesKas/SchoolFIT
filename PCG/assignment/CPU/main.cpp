/*
 * Architektura procesoru (ACH 2016)
 * Projekt c. 1 (nbody)
 * Login: xlogin00
 */

#include <cstdio>

#include "nbody.h"
#include "papi_cntr.h"

int main(int argc, char **argv)
{
    FILE *fp;
    PapiCounterList papi_routines;
    papi_routines.AddRoutine("nbody");

    t_particles particles;

    if (argc != 3)
    {
        printf("Usage: nbody <input> <output>\n");
        exit(1);
    }

    // read particles from file
    fp = fopen(argv[1], "r");
    if (fp == nullptr)
    {
        printf("Can't open file %s!\n", argv[1]);
        exit(1);
    }
    particles_read(fp, particles);
    fclose(fp);

    // print parameters
    printf("N: %d\n", N);
    printf("dt: %f\n", DT);
    printf("steps: %d\n", STEPS);

    // do the measurement
    papi_routines["nbody"].Start();
    particles_simulate(particles);
    papi_routines["nbody"].Stop();

    // write particles to file
    fp = fopen(argv[2], "w");
    if (fp == nullptr)
    {
        printf("Can't open file %s!\n", argv[2]);
        exit(1);
    }
    particles_write(fp, particles);
    fclose(fp);

    // print results
    papi_routines.PrintScreen();

    return 0;
}
