# Architektura procesoru (ACH 2016)
# Projekt c. 1 (nbody)
# Login: xlogin00

CC=icpc
CFLAGS=-std=c++11 -lpapi -ansi-alias
OPT=-O2 -Wall
#REPORT=-qopt-report=5

N=1024
DT=0.001f
STEPS=1000

PARAMS=-DN=$(N) -DDT=$(DT) -DSTEPS=$(STEPS)

INPUT=input.dat
OUTPUT=stepCPU.dat

PAPI_EVENTS=PAPI_SP_OPS

all:
	$(CC) $(CFLAGS) $(OPT) $(REPORT) $(PARAMS) -c velocity.cpp
	$(CC) $(CFLAGS) $(OPT) $(REPORT) $(PARAMS) -c nbody.cpp
	$(CC) $(CFLAGS) $(OPT) $(PARAMS) -S -fsource-asm -c nbody.cpp
	$(CC) $(CFLAGS) $(OPT) $(PARAMS) velocity.o nbody.o main.cpp -o nbody
	$(CC) $(CFLAGS) gen.cpp -o gen

clean:
	rm -f *.o nbody gen

run:
	PAPI_EVENTS='$(PAPI_EVENTS)' ./nbody $(INPUT) $(OUTPUT)

generate:
	./gen 1024 $(INPUT)
