#! /bin/bash
ratio=3.0

t=0.5
dz=500
p_id=0
model_name=vicuna
output_dir='outputs/sum_outputs'
output_dir_log='outputs/sum_outputs/log'
mkdir -p $output_dir
mkdir -p $output_dir_log

python \
  infer_sum.py \
  --conv_type vicuna_sum \
  --model_path /private/model/lmsys/vicuna-7b-v1.5 \
  --principle_id $p_id \
  --temperature $t \
  --ratio $ratio \
  --data_size $dz \
  --output_data_file ${output_dir}/${model_name}_sum_data.json \
  --output_result_file ${output_dir}/${model_name}_opad.json \
  &> ${output_dir_log}/${model_name}_opad.log 
