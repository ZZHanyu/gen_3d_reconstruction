#!/bin/bash

GPU_ID=0
model_list=('xxx')  # your model name

for model in "${model_list[@]}"; do
    CUDA_VISIBLE_DEVICES=${GPU_ID} \
    python tools/testers/infer.py \
        --seed 1234 \
        --checkpoint 'checkpoint/large/model.safetensors' \
        --processing_res 700 \
        --output_dir output/${model} \
        --arch_name 'depthanything-large' # [depthanything-large, depthanything-base, depthanything-small]
done