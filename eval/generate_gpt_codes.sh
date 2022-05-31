#!/bin/bash

name=$1
shift

arch=$1
shift

shard=$1
shift

dir="expts/${name}"
echo $dir

mkdir $dir

python -u generate_gpt_codes.py \
  -t ../data/test.json \
  --train ../data/train.json \
  --arch $arch \
  --save ${dir} \
  -d \
  --shard ${shard} \
  $@ \
  | tee ${dir}/shard-${shard}.out
