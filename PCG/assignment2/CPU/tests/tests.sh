#!/bin/sh

#make (no openMP)
#parameters N DT Steps
MakeSerial () {
	icpc -std=c++11 -lpapi -ansi-alias -O2 -Wall -DN=$1 -DDT=$2 -DSTEPS=$3 -c ../velocity.cpp
	icpc -std=c++11 -lpapi -ansi-alias -O2 -Wall -DN=$1 -DDT=$2 -DSTEPS=$3 -c ../nbody.cpp
	icpc -std=c++11 -lpapi -ansi-alias -O2 -Wall -DN=$1 -DDT=$2 -DSTEPS=$3 velocity.o nbody.o ../main.cpp -o nbody
}

#make (with openMP)
#parameters N DT Steps
MakeVector () {
	icpc -std=c++11 -lpapi -ansi-alias -O2 -Wall -xavx -qopenmp-simd -DN=$1 -DDT=$2 -DSTEPS=$3 -c ../velocity.cpp
	icpc -std=c++11 -lpapi -ansi-alias -O2 -Wall -xavx -qopenmp-simd -DN=$1 -DDT=$2 -DSTEPS=$3 -c ../nbody.cpp
	icpc -std=c++11 -lpapi -ansi-alias -O2 -Wall -xavx -qopenmp-simd -DN=$1 -DDT=$2 -DSTEPS=$3 velocity.o nbody.o ../main.cpp -o nbody
}

#clean files
rm -rf ~test-outputs
mkdir ~test-outputs

#Test: Two particles on circle
echo "Two particles on circular trajectory..."
MakeSerial 2 0.00001f 543847
./nbody ../../test-data/circle.dat ~test-outputs/circle.out >> /dev/null
./test-difference.py ~test-outputs/circle.out ../../test-data/circle-ref.dat


#Test:
echo "Points on line without collision... without vectorization..."
MakeSerial 32 0.001f 10000
./nbody ../../test-data/two-lines.dat ~test-outputs/two-lines.out >> /dev/null
./test-difference.py ~test-outputs/two-lines.out ../../test-data/two-lines-ref.dat

#Test:
echo "Points on line without collision ...with vectorization..."
MakeVector 32 0.001f 10000
./nbody ../../test-data/two-lines.dat ~test-outputs/two-lines-v.out >> /dev/null
./test-difference.py ~test-outputs/two-lines-v.out ../../test-data/two-lines-ref.dat


#Test:
echo "Points on line with one collision... without vectorization..."
MakeSerial 32 0.001f 45000
./nbody ../../test-data/two-lines.dat ~test-outputs/two-lines-one.out >> /dev/null
./test-difference.py ~test-outputs/two-lines-one.out ../../test-data/two-lines-collided-45k.dat

#Test:
echo "Points on line with one collision ...with vectorization..."
MakeVector 32 0.001f 45000
./nbody ../../test-data/two-lines.dat ~test-outputs/two-lines-one-v.out >> /dev/null
./test-difference.py ~test-outputs/two-lines-one-v.out ../../test-data/two-lines-collided-45k.dat


#Test:
echo "Points on line with several collision... without vectorization..."
MakeSerial 32 0.001f 50000
./nbody ../../test-data/two-lines.dat ~test-outputs/two-lines-several.out >> /dev/null
./test-difference.py ~test-outputs/two-lines-several.out ../../test-data/two-lines-collided-50k.dat

#Test:
echo "Points on line with several collision ...with vectorization..."
MakeVector 32 0.001f 50000
./nbody ../../test-data/two-lines.dat ~test-outputs/two-lines-several-v.out >> /dev/null
./test-difference.py ~test-outputs/two-lines-several-v.out ../../test-data/two-lines-collided-50k.dat


#Test:
echo "Light and heavy objects collison 1D ... without vectorization..."
MakeSerial 2 0.01f 505
./nbody ../../test-data/collision.dat ~test-outputs/collision.out >> /dev/null
./test-difference.py ~test-outputs/collision.out ../../test-data/collision-ref.dat

#Test:
echo "Light and heavy objects collison 1D ... with vectorization..."
MakeVector 2 0.01f 505
./nbody ../../test-data/collision.dat ~test-outputs/collision-v.out >> /dev/null
./test-difference.py ~test-outputs/collision-v.out ../../test-data/collision-ref.dat

#Test:
echo "Light and heavy objects collison 2D ... without vectorization..."
MakeSerial 2 0.01f 505
./nbody ../../test-data/collision2.dat ~test-outputs/collision2.out >> /dev/null
./test-difference.py ~test-outputs/collision2.out ../../test-data/collision2-ref.dat

#Test:
echo "Light and heavy objects collison 2D ... with vectorization..."
MakeVector 2 0.01f 505
./nbody ../../test-data/collision2.dat ~test-outputs/collision2-v.out >> /dev/null
./test-difference.py ~test-outputs/collision2-v.out ../../test-data/collision2-ref.dat


#Test:
echo "Symetry globe test...without vectorization..."
MakeSerial 932 0.1f 1
./nbody ../../test-data/thompson_points_932.dat ~test-outputs/thompson.out >> /dev/null
./test-thompson.py ../../test-data/thompson_points_932.dat ~test-outputs/thompson.out


#Test:
echo "Symetry globe test...with vectorization..."
MakeVector 932 0.1f 1
./nbody ../../test-data/thompson_points_932.dat ~test-outputs/thompson-v.out >> /dev/null
./test-thompson.py ../../test-data/thompson_points_932.dat ~test-outputs/thompson-v.out


#Test:
echo "Stability globe test...without vectorization..."
MakeSerial 932 0.00001f 15000
./nbody ../../test-data/thompson_points_932.dat ~test-outputs/thompson.out >> /dev/null
./test-thompson.py ../../test-data/thompson_points_932.dat ~test-outputs/thompson.out


#Test:
echo "Stability globe test...with vectorization..."
MakeVector 932 0.00001f 15000
./nbody ../../test-data/thompson_points_932.dat ~test-outputs/thompson-v.out >> /dev/null
./test-thompson.py ../../test-data/thompson_points_932.dat ~test-outputs/thompson-v.out

rm *.o
