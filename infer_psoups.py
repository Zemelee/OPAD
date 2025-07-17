import argparse

import datasets
from conversation import get_conv_adapter
from utils import *
import os
import random
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import json

from dataset import PSOUPSDataset
from dataset import Principle

import numpy as np

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)

import torch
import torch.nn.functional as F

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--principle_id", type=int)
    parser.add_argument("--conv_type", type=str, default="llama2")
    parser.add_argument(
        "--data_path",
        type=str,
        default="/private/home/zhumingye/code/Linear_Alignment/data/P_soup",
    )
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--output_data_file", type=str, required=True)
    parser.add_argument("--output_result_file", type=str, required=True)
    parser.add_argument("--data_size", type=int, default=20)
    parser.add_argument("--do_sample", action="store_true")
    args = parser.parse_args()
    conv_adapter = get_conv_adapter(args.conv_type)
    principle_list = Principle()
    model_path = args.model_path
    principle = principle_list.principle_list_personal
    generation_config = {
        "max_new_tokens": 512,
        "temperature": args.temperature,
        "top_p": 0.8,
        "do_sample": args.do_sample,
    }
    print("Begin loading dataset !", flush=True)

    raw_dataset = datasets.load_dataset(args.data_path, split="train")
    #######decide number#################
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    print("loading origin model !")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto", torch_dtype=torch.float32
    )

    model = model.eval()
    print("model loading down")
    selected_data = PSOUPSDataset(
        raw_dataset,
        principles=principle,
        conv_adapter=conv_adapter,
        principle_id=args.principle_id,
    )

    data_len = len(selected_data)

    print(f"datasets len: {data_len}")
    generated_data = []

    contra_corr = 0
    principle_corr = 0
    count = 0
    for index, i in tqdm(enumerate(selected_data)):
        print(f"index:{index}", flush=True)
        data_points = i
        no_principle_inputs = tokenizer(
            i["dialog_no_preference"] + "Answer:",
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        no_principle_ids = no_principle_inputs["input_ids"]
        no_principle_att = no_principle_inputs["attention_mask"]

        principle_inputs = tokenizer(
            i["dialog"] + "Answer:", return_tensors="pt", padding=True, truncation=True
        )
        principle_ids = principle_inputs["input_ids"]
        principle_att = principle_inputs["attention_mask"]

        generate_ids_sys = model.generate(principle_ids.cuda(), **generation_config)
        generate_ids_no_principle = model.generate(
            no_principle_ids.cuda(), **generation_config
        )

        init_len = no_principle_ids.shape[1]
        init_len_pref = principle_ids.shape[1]
        principle_output = tokenizer.decode(generate_ids_sys[0, init_len_pref:])
        no_principle_output = tokenizer.decode(generate_ids_no_principle[0, init_len:])

        do_sample = False
        dev = principle_ids.device
        bsz = principle_ids.size(0)

        done = torch.zeros((bsz,), device=dev).to(torch.bool)
        past_key_values_in = None
        past_key_values_no = None
        current_ids = principle_ids
        current_att = principle_att
        current_ids_no = no_principle_ids
        current_att_no = no_principle_att
        output_ids = []
        for i in range(512):  # generate up to 50 tokens
            if done:
                break
            with torch.no_grad():
                if not past_key_values_in:
                    output = model(
                        current_ids,
                        current_att,
                        past_key_values=past_key_values_in,
                        use_cache=True,
                    )
                    logits = output.logits
                    past_key_values_in = output.past_key_values
                    next_token_logit = logits[:, -1, :]
                    next_token_probs = F.softmax(next_token_logit, dim=-1)

                else:
                    logits_old = logits.clone()
                    output = model(
                        next_token_id.unsqueeze(-1),
                        current_att,
                        past_key_values=past_key_values_in,
                        use_cache=True,
                    )
                    logits = output.logits
                    past_key_values_in = output.past_key_values
                    log_posterior_prob = F.log_softmax(
                        logits.mean(1), dim=-1
                    ) + F.log_softmax(logits_old.mean(1), dim=-1)

                    next_token_logit = logits[:, -1, :]

                    logits_no_old = logits_no.clone()
                    output_no = model(
                        next_token_id.unsqueeze(-1),
                        current_att_no,
                        past_key_values=past_key_values_no,
                        use_cache=True,
                    )
                    logits_no = output_no.logits
                    past_key_values_no = output_no.past_key_values
                    log_posterior_prob_no = F.log_softmax(
                        logits_no.mean(1), dim=-1
                    ) + F.log_softmax(logits_no_old.mean(1), dim=-1)
                    next_token_probs = (
                        F.softmax(next_token_logit / 1.0, dim=-1)
                        * (0.5 * (log_posterior_prob - log_posterior_prob_no)).exp()
                    )

            next_token_id = torch.argmax(next_token_probs, dim=-1)
            # Update log-probabilities with the new token
            new_attention_values = torch.ones(
                (current_att.shape[0], 1), dtype=current_att.dtype
            )
            current_att = torch.cat([current_att, new_attention_values], dim=-1)
            current_att_no = torch.cat([current_att_no, new_attention_values], dim=-1)
            current_ids = torch.cat((current_ids, next_token_id.unsqueeze(-1)), dim=1)
            done = done | next_token_id.eq(tokenizer.eos_token_id)

        generated_text = tokenizer.decode(current_ids[0][init_len_pref:])
        data_points["index"] = index
        data_points["no_principle_output"] = no_principle_output
        data_points["principle_output"] = principle_output
        data_points["opad_output"] = generated_text
        generated_data.append(data_points)
        with open(args.output_result_file, "w") as f:
            json.dump(generated_data, f, indent=4)
