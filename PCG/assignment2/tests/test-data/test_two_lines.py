#! /usr/bin/env python3

import sys
import math
import h5py

COUNT = 32
with h5py.File(sys.argv[1], 'r') as f:
    pos_x = f['pos_x_final']
    vel_x = f['vel_x_final']


    if not len(pos_x) == COUNT:
        print("*ERROR* File expects output of simulation using two-lines.dat as input and two-lines-setup.dat as constants in makefile")
        exit(-1)

    fail = False
    positions = []
    #tests if the positions are still simetrical
    for i in range(0,16):
    	#should cancel each other
        diff_pos = pos_x[i] + pos_x[COUNT-i-1]
        diff_vel = vel_x[i] + vel_x[COUNT-i-1]

        if (abs(diff_pos) > 0.001) or (abs(diff_vel) > 0.001):
            fail = True
            positions = [i, COUNT-i-1]

    if fail:
        print("Pairs of particles on " + str(positions[0]) + " and " + str(positions[1]) +" positions do not cancel each other - ERROR")
        print("*ERROR* Points have NOT move uniformly!")
        exit(-1)
    else:
        print("Particles cancel each other: - OK")
        exit(0)
