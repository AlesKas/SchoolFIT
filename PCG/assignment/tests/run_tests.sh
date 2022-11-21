#!/bin/bash

if [[ $# < 1 ]]; then
	echo "Usage: <path to nbody binary>"
	exit -1
fi
#clean files
rm -rf test-outputs
mkdir test-outputs

 
#Test: Two particles on circle
echo "Two particles on circular trajectory"
$1/nbody 2 0.00001f 543847 128 0 32 32 test-data/circle.h5 test-outputs/circle.h5 >> /dev/null
./test-data/test-difference.py test-outputs/circle.h5 test-data/circle-ref.h5

#Test:
echo "Points on line without collision"
$1/nbody 32 0.001f 10000 128 0 32 32 test-data/two-lines.h5 test-outputs/two-lines.h5 >> /dev/null
./test-data/test-difference.py test-outputs/two-lines.h5 test-data/two-lines-ref.h5


#Test:
echo "Points on line with one collision"
$1/nbody 32 0.001f 45000 128 0 32 32 test-data/two-lines.h5 test-outputs/two-lines-one.h5 >> /dev/null
./test-data/test-difference.py test-outputs/two-lines-one.h5 test-data/two-lines-collided-45k.h5


#Test:
echo "Points on line with several collision"
$1/nbody 32 0.001f 50000 128 0 32 32 test-data/two-lines.h5 test-outputs/two-lines-several.h5 >> /dev/null
./test-data/test-difference.py test-outputs/two-lines-several.h5 test-data/two-lines-collided-50k.h5


#Test:
echo "Symetry globe test"
$1/nbody 932 0.1f 1 128 0 32 32 test-data/thompson_points_932.h5 test-outputs/thompson.h5 >> /dev/null
./test-data/test-thompson.py test-outputs/thompson.h5 test-data/thompson_points_932.h5 


#Test:
echo "Stability globe test"
$1/nbody 932 0.00001f 15000 128 0 32 32 test-data/thompson_points_932.h5 test-outputs/thompson.h5 >> /dev/null
./test-data/test-thompson.py test-outputs/thompson.h5 test-data/thompson_points_932.h5
