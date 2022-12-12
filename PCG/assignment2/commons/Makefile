#/**
# * @file      Makefile
# *
# * @author    Jiri Jaros \n
# *            Faculty of Information Technology \n
# *            Brno University of Technology \n
# *            jarosjir@fit.vutbr.cz
# *
# * @brief     PCG Assignment 2
# *            N-Body simulation in ACC - Makefile
# *
# * @version   2022
# *
# * @date      11 November  2020, 11:22 (created) \n
# * @date      04 December  2020, 10:57 (revised) \n
# *
# */

LIBS=-lhdf5

.PHONY: all clean

all: gen

gen: gen.cpp h5Helper.cpp
	g++ gen.cpp h5Helper.cpp ${LIBS} -o gen

clean:
	rm -f *.o gen
