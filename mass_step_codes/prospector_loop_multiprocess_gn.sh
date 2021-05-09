#!/bin/bash

galaxy_seqs=(5823 15153 529 11595 6969 27320 24709 28563 
             24182 9831 18260 1476 21156 10988 7360 12525 
             16374 9381 10555 4278 8478 5545 23694 16769 
             12327 4504 17904 20090 27438 6287 7220 8940 
             4658 13556)

ncores=7

batch=0

num_galaxies=${#galaxy_seqs[@]}
total_batches=$((num_galaxies/ncores))

echo "Starting at:" $(date)
echo "Will run $total_batches batches on $ncores cores for $num_galaxies galaxies." 

for ((batch=0; batch<=$total_batches; batch++))
do
    init=$((batch*ncores))
    for (( c=$init; c<$init+$ncores; c++ ))
    do
        if [ $c -lt $num_galaxies ]
        then
            echo "$batch $init $c    python prospector_goods_fit.py North ${galaxy_seqs[c]} &"
            python prospector_goods_fit.py 'North' ${galaxy_seqs[c]} &
            # comment out above line to do a dry run
        fi
    done
wait
echo "-------- Finished batch $batch"
done

echo "Finished at:" $(date)
