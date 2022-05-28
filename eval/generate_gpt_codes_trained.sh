#!/bin/bash

arch=$1
load=$2
k=$3

dir="expts/`basename $arch`-k${k}"
echo $dir

mkdir dir

python -u generate_gpt_codes.py \
  -t ../data/test.json \
  --train ../data/train.json \
  --arch $arch \
  --save ${dir} \
  -d \
  --start 0 \
  --end 1000 \
  | tee ${dir}.out
