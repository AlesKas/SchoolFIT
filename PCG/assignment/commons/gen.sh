for i in {10..25}
do
    let a=5*$i*512
    ./gen $a ../data/$a.h5
done