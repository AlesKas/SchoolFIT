for i in {1..10}
do
    let a=$i*1024
    ./commons/gen $a data/$a.h5
    ./step0/nbody $a 0.01f 1 512 20 4096 128 data/$a.h5 data/$a.output.h5
    ./step1/nbody $a 0.01f 1 512 20 4096 128 data/$a.h5 data/$a.output.h5
    ./step2/nbody $a 0.01f 1 512 20 4096 128 data/$a.h5 data/$a.output.h5
    echo ""
done