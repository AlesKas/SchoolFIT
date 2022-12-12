#! /usr/bin/env python3

import sys
import math
import h5py

start_pos = []
end_pos   = []

with h5py.File(sys.argv[1], 'r') as f:
    pos_x = f['pos_x_final']
    pos_y = f['pos_y_final']
    pos_z = f['pos_z_final']
    
    start_pos = [list(entry) for entry in zip(pos_x[...], pos_y[...], pos_z[...])]

with h5py.File(sys.argv[2], 'r') as f2:
    pos_x = f2['pos_x_final']
    pos_y = f2['pos_y_final']
    pos_z = f2['pos_z_final']
    
    end_pos = [list(entry) for entry in zip(pos_x[...], pos_y[...], pos_z[...])]

if not len(start_pos) == len(end_pos):
    print ("*ERROR* Start and final clouds have different number of points!")
    exit(-1)

min_dist = sys.float_info.max
max_dist = -sys.float_info.max
for i in range(0,len(start_pos)):
    delta_pos = []
    for j in range(0,3):
        delta_pos.append(start_pos[i][j] - end_pos[i][j])
    
    dist = 0.0
    for j in range(0,3):
        dist = dist + delta_pos[j] * delta_pos[j]
    
    dist = math.sqrt(dist)
    if min_dist > dist:
        min_dist = dist
    
    if max_dist < dist:
        max_dist = dist

print("Maximum distance: " + str(max_dist))
print("Minimum distance: " + str(min_dist))

if abs(max_dist - min_dist) > 0.01:
    print("Distance difference: " + str(abs(max_dist - min_dist)) + " > 0.01 - ERROR")
    print("*ERROR* Points have NOT move uniformly!")
else:
    print("Distance difference: " + str(abs(max_dist - min_dist)) + " < 0.01 - OK")

min_dot = sys.float_info.max
max_dot = -sys.float_info.max

for i in range(0,len(start_pos)):
    start_vec = start_pos[i]
    end_vec   = end_pos[i]
    
    vec_len = math.sqrt(start_vec[0] * start_vec[0] + 
                        start_vec[1] * start_vec[1] + 
                        start_vec[2] * start_vec[2])
    start_vec[0] = start_vec[0] / vec_len
    start_vec[1] = start_vec[1] / vec_len
    start_vec[2] = start_vec[2] / vec_len
    
    vec_len = math.sqrt(end_vec[0] * end_vec[0] + 
                        end_vec[1] * end_vec[1] + 
                        end_vec[2] * end_vec[2])
    end_vec[0] = end_vec[0] / vec_len
    end_vec[1] = end_vec[1] / vec_len
    end_vec[2] = end_vec[2] / vec_len
    
    dot = start_vec[0] * end_vec[0] + start_vec[1] * end_vec[1] + start_vec[2] * end_vec[2]
    
    if min_dot > dot:
        min_dot = dot
        
    if max_dot < dot:
        max_dot = dot

print("")
print("Maximum angle: " + str(math.degrees(math.acos(max(-1.0, min_dot)))) + " deg")
print("Minimum angle: " + str(math.degrees(math.acos(min(1.0, max_dot))))  + " deg")

if math.degrees(math.acos(max(-1.0, min_dot))) > 1.0:
    print("*ERROR* Points haven NOT move uniformly towards center of mass!")
    print("Maximum angle is more than 1* (should be as close to 0 as possible).")
else:
    print("OK")
