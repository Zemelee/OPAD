#! /bin/bash
ratio=3.0

t=0.5
dz=400
p_id=0
model_name=vicuna
output_dir='outputs/hh_outputs'
output_dir_log='outputs/hh_outputs/log'
mkdir -p $output_dir
mkdir -p $output_dir_log

python \
  infer_hh_posterior.py \
  --conv_type vicuna \
  --model_path /private/model/lmsys/vicuna-7b-v1.5 \
  --principle_id $p_id \
  --temperature $t \
  --ratio $ratio \
  --data_size $dz \
  --output_data_file ${output_dir}/${model_name}_hh_data.json \
  --output_result_file ${output_dir}/${model_name}_opad.json \
  &> ${output_dir_log}/${model_name}_opad.log 
