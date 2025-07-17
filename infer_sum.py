import argparse

from conversation import get_conv_adapter
from utils import *

import random
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import datasets
import time

from dataset import CDDataset, SUMDataset
from model import ConstractiveDecodingModel

from dataset import Principle
import numpy as np
import os

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)


os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"


def kl_divergence_easy(logp, logq):
    kl = torch.sum((logp.exp() * (logp - logq)))
    return kl.item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--principle_id", type=int)
    parser.add_argument("--conv_type", type=str, default="llama2")
    parser.add_argument("--data_path", type=str, default="Anthropic/hh-rlhf")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--output_data_file", type=str, required=True)
    parser.add_argument("--output_result_file", type=str, required=True)
    parser.add_argument("--data_size", type=int, default=20) # 测试数据量
    parser.add_argument("--do_sample", action="store_true")
    args = parser.parse_args()
    conv_adapter = get_conv_adapter(args.conv_type)
    model_path = args.model_path

    generation_config = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "do_sample": args.do_sample,
    }
    print("Loading dataset !", flush=True)
    raw_dataset = datasets.load_dataset(
        "/private/home/zhumingye/data/data/pro_sum_data/summarize_test", split="test"
    )

    shuffled_dataset = raw_dataset.shuffle(seed=42)

    sampled_dataset = shuffled_dataset.select(range(args.data_size))

    del raw_dataset, shuffled_dataset
    print("Dataset loaded !", flush=True)

    if "qwen" in model_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            pad_token="<|im_end|>",
            eos_token="<|im_end|>",
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading origin model !")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto", torch_dtype=torch.float16
    )

    model = model.eval()
    print("Model loaded!")

    selected_data = SUMDataset(sampled_dataset, conv_adapter=conv_adapter)

    sampled_dataset.to_json(args.output_data_file)

    data_len = len(selected_data)

    print(f"datasets len: {data_len}")

    generated_data = []

    for index, i in tqdm(enumerate(selected_data)):
        principle_text = i["dialogue_text_principle"]
        no_principle_text = i["dialogue_text"]
        chosen_answer = i["answer"]
        principle = i["principle"]
        question = i["question"]

        inputs1 = tokenizer(principle_text, return_tensors="pt")  ####for likelihood
        ids1 = inputs1["input_ids"]
        att1 = inputs1["attention_mask"]
        inputs2 = tokenizer(
            no_principle_text, return_tensors="pt"
        )  ###就是question, for prior
        ids2 = inputs2["input_ids"]
        att2 = inputs2["attention_mask"]

        len_principal = len(ids1[0])
        len_no_principal = len(ids2[0])

        inputs = no_principle_text

        generate_ids2 = model.generate(ids1.cuda(), **generation_config)

        principal_output = tokenizer.decode(generate_ids2[0][len_principal:])

        past_key_values_in = None
        past_key_values_no = None
        current_ids = ids1
        current_att = att1
        current_att_no = att2
        current_ids_no = ids2
        output_ids = []

        do_sample = False
        dev = ids1.device
        bsz = ids1.size(0)

        done = torch.zeros((bsz,), device=dev).to(torch.bool)

        log_probs = []
        kl_tokens = []

        for i in range(args.max_new_tokens):  # generate up to 50 tokens
            if done.all():
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

                    output_no = model(
                        current_ids_no,
                        current_att_no,
                        past_key_values=past_key_values_no,
                        use_cache=True,
                    )
                    logits_no = output_no.logits
                    past_key_values_no = output_no.past_key_values

                    next_token_logit = logits[:, -1, :]

                    next_token_probs = F.softmax(
                        next_token_logit / 1.0, dim=-1
                    )  ###*torch.exp(neg_energy) ###*(F.softmax(log_posterior_prob,dim=-1))

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
                    next_token_logit = logits[:, -1, :]  ######

                    neg_energy = 1.0 * (log_posterior_prob - log_posterior_prob_no)
                    next_token_probs = F.softmax(
                        next_token_logit / 1.0, dim=-1
                    ) * torch.exp(neg_energy)
                    next_token_probs = next_token_probs / next_token_probs.sum(
                        dim=-1, keepdim=True
                    )

                next_token_id = torch.argmax(next_token_probs, dim=-1)

            new_attention_values = torch.ones(
                (current_att.shape[0], 1), dtype=current_att.dtype
            )
            current_att = torch.cat([current_att, new_attention_values], dim=-1)
            current_att_no = torch.cat([current_att_no, new_attention_values], dim=-1)
            current_ids = torch.cat((current_ids, next_token_id.unsqueeze(-1)), dim=1)

            done = done | next_token_id.eq(tokenizer.eos_token_id)

        generated_text = tokenizer.decode(current_ids[0][len_principal:])

        data_points = {
            "id": index,
            "question": question,
            "principal_output": principal_output,
            "modified_output": generated_text,
        }

        generated_data.append(data_points)

        with open(args.output_result_file, "w") as f:
            json.dump(generated_data, f, indent=4)
