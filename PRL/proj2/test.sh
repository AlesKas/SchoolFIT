#!/bin/bash

# Počet uzlů
NUM_NODES=`expr length "$1"`

# Počet procesorů nutný k vytvoření Eulerovy cesty
NUM_PROCESSORS=$((2*NUM_NODES - 2))

mpic++ --prefix /usr/local/share/OpenMPI -o main main.cpp

mpirun --prefix /usr/local/share/OpenMPI -oversubscribe -np $NUM_PROCESSORS  main $1 $NUM_NODES

rm main