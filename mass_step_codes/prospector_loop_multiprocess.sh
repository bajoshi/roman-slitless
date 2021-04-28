#!/bin/bash

galaxy_seqs=(12050 14251 3689 8859 7837 18603 16622 
    15591 11855 19962 19729 25920 20219 25895 16531 
    2802 24093 12984 15316 965 1244 8736 24539 4997 
    3456 6807 6229 9726 20479 18904 20227 10565)

ncores=6

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
            echo "$batch $init $c    python prospector_goods_fit.py ${galaxy_seqs[c]} &"
            python prospector_goods_fit.py ${galaxy_seqs[c]} &  # comment out this line to do a dry run
        fi
    done
wait
echo "-------- Finished batch $batch"
done

echo "Finished at:" $(date)
