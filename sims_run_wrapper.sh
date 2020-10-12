#!/bin/csh

python /Users/baj/Documents/GitHub/roman-slitless/gen_sed_lst.py
wait
echo "Finished with generating sed lst."

python /Users/baj/Documents/GitHub/roman-slitless/run_pylinear.py
wait

echo "Finshed with pylinear simulation and extraction."