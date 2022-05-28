#!/bin/bash

for f in `squeue -u $USER -S i | grep train | awk '{print $1}'`
do 
  echo -n $f
  grep save_dir slurm-${f}.out | head -n1
done
# ujobs_long: aliased to squeue -u $USER -S i 
