# Paralelní programování na GPU (PCG 2021)
# Projekt c. 1 (cuda)
# Login: xlogin00

LIBS=-lhdf5

.PHONY: all clean

all: gen

gen: gen.cpp h5Helper.cpp
	g++ gen.cpp h5Helper.cpp ${LIBS} -o gen

clean:
	rm -f *.o nbody
