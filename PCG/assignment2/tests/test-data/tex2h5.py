#! /usr/bin/env python3
import sys
import h5py
import numpy as np
import re

if len(sys.argv) < 3:
    print("Usage: <input file> <output file>")
    exit(-1)

with open(sys.argv[1]) as f:
    content = f.readlines()


content_size = len(content)

print(len(content))



pos_x = []
pos_y = []
pos_z = []

vel_x = []
vel_y = []
vel_z = []

weight = []

i = 0
for line in content:
    line = line.rstrip('\n').lstrip()  # Remove EOF from line
    columns = re.compile(r"\s+").split(line)  # Get the data groups 


    pos_x .append(float(columns[0]))
    pos_y .append(float(columns[1]))
    pos_z .append(float(columns[2]))
    vel_x .append(float(columns[3]))
    vel_y .append(float(columns[4]))
    vel_z .append(float(columns[5]))
    weight.append(float(columns[6]))
    i += 1

output  = h5py.File(sys.argv[2], 'w')
d_pos_x = output.create_dataset("pos_x", data = pos_x)
d_pos_y = output.create_dataset("pos_y", data = pos_y)
d_pos_z = output.create_dataset("pos_z", data = pos_z)

d_vel_x = output.create_dataset("vel_x", data = vel_x)
d_vel_y = output.create_dataset("vel_y", data = vel_y)
d_vel_z = output.create_dataset("vel_z", data = vel_z)

d_weight = output.create_dataset("weight", data = weight)

output.close()