for i in {10..25}
do
    let a=5*$i*512
    ./nbody $a 0.01f 1 512 20 4096 128 ../data/$a.h5 ../data/$a.output.h5
done