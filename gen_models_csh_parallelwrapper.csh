#!/bin/csh -f

set ncores = 8
set total_templates = 70
@ total_par_runs = $total_templates / $ncores

echo "Beginning code at:" `date`
echo "Will run code on:" $ncores "cores"
echo "Total parallel runs:" $total_par_runs

set counter = 131
# counter is where you wnat to start the tau value from

foreach i (`seq 0 $total_par_runs`)
    @ jend = $i + $ncores - 1
    foreach j (`seq $i $jend`)
        python $argv[1] $counter &
        #echo $i $j $counter
        @ counter++
    end
    wait
end

echo "All done." `date`