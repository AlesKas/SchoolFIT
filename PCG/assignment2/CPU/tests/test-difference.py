#!/usr/bin/python

import sys
import math

start_pos = []
end_pos   = []

with open(sys.argv[1], 'r') as f:
    for line in f:
        vals = line.split()
        start_pos.append([ float(vals[0]), float(vals[1]), float(vals[2]) ])

with open(sys.argv[2], 'r') as f:
    for line in f:
        vals = line.split()
        end_pos.append([ float(vals[0]), float(vals[1]), float(vals[2]) ])

if not len(start_pos) == len(end_pos):
    print "*ERROR* Start and final clouds have different number of points!"
    exit(-1)

peak_diff = 0

for i in range(0,len(start_pos)):
    delta_pos = []
    for j in range(0,3):
        diff = abs(start_pos[i][j] - end_pos[i][j])
	if diff > peak_diff:
		peak_diff = diff    	

print("Maximum in possition difference:  " + str(peak_diff))

if peak_diff > 0.01:
    print("*ERROR* The simulation output does not match the expected result")
else:
    print("*OK*")

