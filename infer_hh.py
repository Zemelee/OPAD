import argparse
import seaborn as sns
from conversation import get_conv_adapter
from utils import *

import random
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import datasets
from dataset import CDDataset
from model import ConstractiveDecodingModel
import os
from dataset import Principle
import numpy as np
import matplotlib.pyplot as plt
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)


import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--principle_id",
                        type=int)

    parser.add_argument("--conv_type",
                        type=str,
                        default="llama2")

    parser.add_argument("--data_path",
                        type=str,
                        default='Anthropic/hh-rlhf')

    parser.add_argument("--model_path",
                        type=str,
                        default=None)

    parser.add_argument("--temperature",
                        type=float,
                        default=1.0)

    parser.add_argument("--top_p",
                        type=float,
                        default=0.8)

    parser.add_argument("--max_new_tokens",
                        type=int,
                        default=512)

    parser.add_argument("--output_data_file",
                        type=str,
                        required=True)

    parser.add_argument("--output_result_file",
                        type=str,
                        required=True)

    parser.add_argument("--data_size",
                        type=int,
                        default=1000)

    parser.add_argument("--ratio",
                        type=float,
                        default=2.0)

    parser.add_argument("--do_sample",
                        action="store_true")

    args = parser.parse_args()

    conv_adapter = get_conv_adapter(args.conv_type)

    principle_list = Principle()
    model_path = args.model_path
    principle = principle_list.principle_list_hh[args.principle_id]

    generation_config = {
        'max_new_tokens': args.max_new_tokens,
        'temperature': args.temperature,
        "top_p": args.top_p,
        "do_sample": False
    }

    cd_config = {
        "ratio": args.ratio
    }

    print("Loading dataset !", flush=True)
    raw_dataset = datasets.load_dataset("/private/home/zhumingye/data/data/hh-rlhf", split='test')

    shuffled_dataset = raw_dataset.shuffle(seed=42)

    sampled_dataset = shuffled_dataset.select(range(args.data_size))

    del raw_dataset, shuffled_dataset
    print('Dataset loaded !', flush=True)

    if "qwen" in model_path:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, pad_token='<|im_end|>',
                                                  eos_token='<|im_end|>')
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    print('Loading origin model !')
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 device_map='auto',
                                                 torch_dtype=torch.float32)

    model = model.eval()
    print('Model loaded!')

    selected_data = CDDataset(sampled_dataset, principle=principle, conv_adapter=conv_adapter)


 
    sampled_dataset.to_json(args.output_data_file)

    data_len = len(selected_data)

    print(f"datasets len: {data_len}")

    generated_data = []

    principle = selected_data.principle
    inputs3 = tokenizer(principle, return_tensors='pt')
    ids3 = inputs3['input_ids']
    att3 = inputs3['attention_mask']


    for index, i in tqdm(enumerate(selected_data)):
        print(f"index:{index}", flush=True)

        principle_text = i["dialogue_text_principle"]
        no_principle_text = i["dialogue_text"]
        question = i['question']
        chosen_answer = i["chosen_answer"]
        
        inputs1 = tokenizer(principle_text, return_tensors='pt') ####for likelihood
        ids1 = inputs1['input_ids']
        att1 = inputs1['attention_mask']
        inputs2 = tokenizer(no_principle_text, return_tensors='pt') ###就是question, for prior
        ids2 = inputs2['input_ids']
        att2 = inputs2['attention_mask']

        len_principal = len(ids1[0])
        len_no_principal = len(ids2[0])

        inputs = principle_text

        generate_ids1 = model.generate(ids1.cuda(), **generation_config)

        generate_ids2 = model.generate(ids2.cuda(), **generation_config)

        principal_output = tokenizer.decode(generate_ids1[0][len_principal:])
        sft_output = tokenizer.decode(generate_ids2[0][len_no_principal:])
        

        past_key_values_in = None
        past_key_values_no=None
        current_ids = ids1.cuda()
        current_att = att1.cuda()
        current_att_no=att2.cuda()
        current_ids_no=ids2.cuda()
        output_ids = []

        do_sample=False
        dev = model.device
        bsz = ids1.size(0)
    
        done = torch.zeros((bsz,), device=dev).to(torch.bool)

        kl_tokens=[]

        original_prob=[]
        modified_prob=[]
        neg_energys=[]
 
        for i in range(args.max_new_tokens):  
            if done.all():
                break
            with torch.no_grad():
                if not past_key_values_in: 
                    output = model(current_ids,current_att,past_key_values=past_key_values_in,use_cache=True)
                    logits = output.logits
                    past_key_values_in=output.past_key_values
                    
                    output_no = model(current_ids_no,current_att_no,past_key_values=past_key_values_no,use_cache=True)
                    logits_no = output_no.logits
                    past_key_values_no=output_no.past_key_values

                    next_token_logit = logits[:,-1, :]

                    next_token_probs = F.softmax(next_token_logit/1.0, dim=-1)

                else:
                    logits_old =logits.clone()
                    output = model(next_token_id.unsqueeze(-1),current_att,past_key_values=past_key_values_in,use_cache=True)
                    logits = output.logits
                    past_key_values_in=output.past_key_values
                    log_prob =  F.log_softmax(logits.mean(1),dim=-1)+F.log_softmax(logits_old.mean(1),dim=-1)

                    logits_no_old=logits_no.clone()
                    output_no = model(next_token_id.unsqueeze(-1),current_att_no,past_key_values=past_key_values_no,use_cache=True)
                    logits_no = output_no.logits
                    past_key_values_no=output_no.past_key_values
                    log_prob_no =  F.log_softmax(logits_no.mean(1),dim=-1)+F.log_softmax(logits_no_old.mean(1),dim=-1)

                    next_token_logit = logits[:, -1, :]######

                    neg_energy = (1.0*(log_prob-log_prob_no))
                    next_token_probs = F.softmax(next_token_logit/1.0, dim=-1)*torch.exp(neg_energy) 
                    next_token_probs = next_token_probs/next_token_probs.sum(dim=-1, keepdim=True)


            next_token_id = torch.argmax(next_token_probs, dim=-1)

            new_attention_values = torch.ones((current_att.shape[0], 1), dtype=current_att.dtype,device = dev)
            current_att = torch.cat([current_att, new_attention_values], dim=-1)
            current_att_no = torch.cat([current_att_no, new_attention_values], dim=-1)
            current_ids = torch.cat((current_ids, next_token_id.unsqueeze(-1)), dim=1)
  
            done = done | next_token_id.eq(tokenizer.eos_token_id)

        generated_text = tokenizer.decode(current_ids[0][len_principal:])

   

        data_points = {
            "id": index,
            "inputs": principle_text,
            "principal": principle,
            "sft_output": sft_output,
            "principal_output": principal_output,
            "modified_output": generated_text,
        }
        generated_data.append(data_points)

        with open(args.output_result_file, 'w') as f:
            json.dump(generated_data, f, indent=4)
