#!/bin/bash

name=$1

arch=$2
if [ -z $arch ]
then
  arch="EleutherAI/gpt-neo-2.7B"
fi

save_dir="expts/${name}"

mkdir $save_dir

USE_TF=NO deepspeed tune_apps_gpt.py  \
  --save-dir=${save_dir}  \
  --arch=${arch} \
  --apps-train-files ../data/train \
  --apps-dataroot ../data/train/ \
  --grad-acc-steps=8 \
  --epochs=10 \
  --fp16 \
  --deepspeed deepspeed_config.json \
  --batch-size-per-replica=8 \
  --gradient-checkpointing \
  | tee ${save_dir}/log.out
