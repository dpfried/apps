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

save_dir="/checkpoint/dpf/apps/expts/${name}"

mkdir $save_dir

CUDA_VISIBLE_DEVICES=0 USE_TF=NO python tune_apps_gpt.py  \
  --save-dir=${save_dir}  \
  --arch=${arch} \
  --apps-train-files ../data/train \
  --apps-dataroot ../data/train/ \
  --grad-acc-steps=1 \
  --batch-size-per-replica=1 \
  --epochs=10 \
  --fp16 \
  --gradient-checkpointing \
  $@ \
  | tee ${save_dir}/log.out
