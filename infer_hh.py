import argparse
from conversation import get_conv_adapter
from utils import *

import random
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import json

import numpy as np
import datasets
from dataset import CDDataset
import os
from dataset import Principle
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--principle_id", type=int, default=0)
    parser.add_argument("--conv_type", type=str, default="llama2")
    parser.add_argument("--data_path", type=str, default="Anthropic/hh-rlhf")
    parser.add_argument(
        "--model_path", type=str, default="mistralai/Mistral-7B-Instruct-v0.1"
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--max_new_tokens", type=int, default=800)
    parser.add_argument("--output_data_file", type=str, default="outputs/hh/data.json")
    parser.add_argument("--output_file", type=str, default="outputs/hh/opad.json")
    parser.add_argument("--data_size", type=int, default=3)  # 测试数据量
    parser.add_argument("--do_sample", action="store_true")
    args = parser.parse_args()

    # system 你是有用的ai，{} |user {}
    conv_adapter = get_conv_adapter(args.conv_type)
    principle_list = Principle()
    model_path = args.model_path
    # "请遵循以下原则：尽可能避免事实错误..."
    principle = principle_list.principle_list_hh[args.principle_id]
    generation_config = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "do_sample": False,
    }

    raw_dataset = datasets.load_dataset(args.data_path, split="test")
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
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto", load_in_4bit=True
    )
    model = model.eval()
    print("Model loaded!")
    # sampled_dataset: 指定数量的打乱后的数据集
    # principle: 指定原则
    # conv_adapter: 对话适配器 你是有用的ai{}|user{}
    selected_data = CDDataset(
        sampled_dataset, principle=principle, conv_adapter=conv_adapter
    )
    sampled_dataset.to_json(args.output_data_file)
    data_len = len(selected_data)
    print(f"datasets len: {data_len}")

    generated_data = []
    principle = selected_data.principle
    inputs3 = tokenizer(principle, return_tensors="pt")
    ids3 = inputs3["input_ids"]  # 遵循原则，危险问题提供解释和引导
    att3 = inputs3["attention_mask"]
    for index, i in tqdm(enumerate(selected_data)):
        print(f"index/data_len:{index+1}/{data_len}", "-" * 50, flush=True)
        # i: question dialogue_text dialogue_text_principle chosen_answer reject_answer
        principle_text = i["dialogue_text_principle"]  # 原则 human,AI,human?
        no_principle_text = i["dialogue_text"]  # 不带原则的 human,AI,human?
        question = i["question"]  # 第二个问题
        chosen_answer = i["chosen_answer"]
        # 分词处理
        inputs1 = tokenizer(principle_text, return_tensors="pt")
        ids1 = inputs1["input_ids"]
        att1 = inputs1["attention_mask"]
        len_principal = len(ids1[0])  # 300+
        inputs2 = tokenizer(no_principle_text, return_tensors="pt")
        ids2 = inputs2["input_ids"]
        att2 = inputs2["attention_mask"]
        len_no_principal = len(ids2[0])  # 200+
        # 有原则引导的生成
        generate_ids1 = model.generate(ids1.cuda(), **generation_config)  # 有原则q
        generate_ids2 = model.generate(ids2.cuda(), **generation_config)  # 无原则q
        # 去除前缀只保留答案
        principal_output = tokenizer.decode(generate_ids1[0][len_principal:])
        sft_output = tokenizer.decode(generate_ids2[0][len_no_principal:])
        # 改进后的生成
        current_ids, current_att = ids1.cuda(), att1.cuda()  # 有原则
        current_ids_no, current_att_no = ids2.cuda(), att2.cuda()  # 无原则
        output_ids = []
        do_sample = False
        dev = model.device
        bsz = ids1.size(0)
        # done 标记每个样本是否已经生成到 EOS
        done = torch.zeros((bsz,), device=dev).to(torch.bool)
        kl_tokens = []
        original_prob, modified_prob = [], []
        neg_energys = []
        past_key_values_in, past_key_values_no = None, None
        # 逐 token 生成文本
        for i in range(args.max_new_tokens):
            if done.all():
                break
            with torch.no_grad():
                if not past_key_values_in:
                    # 有原则，单个token生成
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
                    )  # 无原则
                    logits_no = output_no.logits
                    past_key_values_no = output_no.past_key_values
                    # 取最后一个 token 的输出
                    next_token_logit = logits[:, -1, :]
                    # 转成概率
                    next_probs = F.softmax(next_token_logit / 1.0, dim=-1)
                else:
                    # logits.shape = [bs, seq_len, vocab_size]
                    # mean(1)对第一个维度取平均
                    # dim=-1对最后一个维度操作
                    # unsqueeze(-1) 在最后一维增加一个维度
                    # # 计算带原则的输出=============
                    logits_old = logits.clone()
                    output = model(
                        # inputs 要求[bs, seq_len]
                        next_token_id.unsqueeze(-1),
                        current_att,
                        past_key_values=past_key_values_in,
                        use_cache=True,
                    )
                    logits = output.logits  # 新的有原则预测
                    past_key_values_in = output.past_key_values
                    # 计算当前和历史的 带原则 对数概率
                    log_prob1 = F.log_softmax(logits.mean(1), dim=-1)
                    log_prob2 = F.log_softmax(logits_old.mean(1), dim=-1)
                    # 累加历史信息
                    log_prob = log_prob1 + log_prob2
                    # 计算不带原则的输出=============
                    logits_no_old = logits_no.clone()
                    output_no = model(
                        next_token_id.unsqueeze(-1),
                        current_att_no,
                        past_key_values=past_key_values_no,
                        use_cache=True,
                    )
                    logits_no = output_no.logits  # 不带原则的下一个token
                    past_key_values_no = output_no.past_key_values
                    # 计算不带原则的“当前+历史”联合对数概率
                    log_prob1 = F.log_softmax(logits_no.mean(1), dim=-1)
                    log_prob2 = F.log_softmax(logits_no_old.mean(1), dim=-1)
                    
                    log_prob_no = log_prob1 + log_prob2  # 无原则模型的“当前”预测
                    next_token_logit = logits[:, -1, :] # [bs, seq_len, vocab_size]
                    # 计算“带原则”与“不带原则”的概率差
                    neg_energy = 1.0 * (log_prob - log_prob_no) # 1.0及超参
                    # reward 调整生成概率
                    next_probs = F.softmax(next_token_logit / 1.0, dim=-1) 
                    next_probs = next_probs* torch.exp(neg_energy)
                    # 归一化，相当于应用1/z
                    next_probs = next_probs / next_probs.sum(dim=-1, keepdim=True)
            # 选择概率最大的 toke作为下一个token
            next_token_id = torch.argmax(next_probs, dim=-1)
            # 新增token的掩码为1，表示“需要被关注”
            new_attention_values = torch.ones(
                (current_att.shape[0], 1), dtype=current_att.dtype, device=dev
            )
            # 带原则的掩码更新
            current_att = torch.cat([current_att, new_attention_values], dim=-1)
            current_att_no = torch.cat([current_att_no, new_attention_values], dim=-1)
            current_ids = torch.cat((current_ids, next_token_id.unsqueeze(-1)), dim=1)
            # 如果生成了终止token（eos_token），标记为完成
            done = done | next_token_id.eq(tokenizer.eos_token_id)

        generated_text = tokenizer.decode(current_ids[0][len_principal:])
        # 保存结果
        data_points = {
            "id": index,
            "inputs": principle_text,
            "principal": principle,
            "sft_output": sft_output,
            "principal_output": principal_output,
            "modified_output": generated_text,
        }
        generated_data.append(data_points)
        with open(args.output_file, "w") as f:
            json.dump(generated_data, f, indent=4)
