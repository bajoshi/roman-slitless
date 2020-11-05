#!/bin/csh -f

set ncores = 6
set total_templates = 20000
@ total_par_runs = $total_templates / $ncores

echo "Beginning code at:" `date`
echo "Will run code on:" $ncores "cores"

set counter = 0

foreach i (`seq 0 $total_par_runs`)
    @ jend = $i + $ncores - 1
    foreach j (`seq $i $jend`)
        python $argv[1] $counter &
        @ counter++
    end
    wait
end

echo "All done." `date`