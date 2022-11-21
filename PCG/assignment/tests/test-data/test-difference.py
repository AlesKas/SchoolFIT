#! /usr/bin/env python3

import sys
import math
import h5py

start_pos = []
end_pos   = []

with h5py.File(sys.argv[1], 'r') as output:
    pos_x = output['pos_x_final']
    pos_y = output['pos_y_final']
    pos_z = output['pos_z_final']
    
    start_pos = [list(entry) for entry in zip(pos_x[...], pos_y[...], pos_z[...])]

with h5py.File(sys.argv[2], 'r') as ref:
    pos_x = ref['pos_x']
    pos_y = ref['pos_y']
    pos_z = ref['pos_z']
    
    end_pos = [list(entry) for entry in zip(pos_x[...], pos_y[...], pos_z[...])]

if not len(start_pos) == len(end_pos):
    print("*ERROR* Start and final clouds have different number of points!")
    exit(-1)

peak_diff = 0

for i in range(0,len(start_pos)):
    delta_pos = []
    for j in range(0,3):
        diff = abs(start_pos[i][j] - end_pos[i][j])
    if diff > peak_diff:
        peak_diff = diff        

print("Peak in possition difference:  " + str(peak_diff))

if peak_diff > 0.01:
    print("*ERROR* The simulation output does not match the expected result")
else:
    print("OK")

