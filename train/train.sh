#!/bin/bash

name=$1
shift

arch=$1
if [ -z $arch ]
then
  arch="EleutherAI/gpt-neo-2.7B"
else
  shift
fi

save_dir="expts/${name}"

mkdir $save_dir

# we run this on 8 GPUs, so the total batch size is 8 x 4 x 8 = 256
USE_TF=NO deepspeed tune_apps_gpt.py  \
  --save-dir=${save_dir}  \
  --arch=${arch} \
  --apps-train-files ../data/train \
  --apps-dataroot ../data/train/ \
  --grad-acc-steps=4 \
  --batch-size-per-replica=8 \
  --epochs=10 \
  --fp16 \
  --deepspeed deepspeed_config.json \
  --gradient-checkpointing \
  $@ \
  | tee ${save_dir}/log.out
