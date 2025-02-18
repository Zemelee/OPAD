#! /bin/bash
ratio=5.0
t=0.5
dz=500
p_id=0

model_name=vicuna
output_dir='outputs/dsp_outputs'
output_log_dir='outputs/dsp_outputs/log'
mkdir -p $output_dir
mkdir -p $output_log_dir

python \
  infer_dsp.py \
  --conv_type vicuna \  ####change it to llama2 if use mistrial model
  --model_path /private/model/lmsys/vicuna-7b-v1.5 \
  --principle_id $p_id \
  --temperature $t \
  --ratio $ratio \
  --data_size $dz \
  --output_data_file ${output_dir}/dsp_data.json \
  --output_result_file ${output_dir}/opad.json \
  &> ${output_log_dir}/${model_name}_opad.log
