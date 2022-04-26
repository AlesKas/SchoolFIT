#!/bin/bash

# Počet uzlů
NUM_NODES=`expr length "$1"`

# Počet procesorů nutný k vytvoření Eulerovy cesty
NUM_PROCESSORS=$((2*NUM_NODES - 2))

mpic++ --prefix /usr/local/share/OpenMPI -o pro pro.cpp

mpirun --prefix /usr/local/share/OpenMPI -oversubscribe -np $NUM_PROCESSORS  pro $1 $NUM_NODES

rm pro