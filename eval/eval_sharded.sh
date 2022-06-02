#!/bin/bash

save_dir=$1
shard=$2

python test_one_solution.py \
  --shard ${shard} \
  --save ${save_dir} \
  --test_loc ../data/test.json \
  | tee ${save_dir}/eval-${shard}.out
