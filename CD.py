import argparse

from conversation import get_conv_adapter
from utils import *
import loguru
import random
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import datasets

from dataset import CDDataset
import os
from dataset import Principle
import numpy as np

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)

os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

def parse_args():
    parser = argparse.ArgumentParser(
        description=
        "Finetune a transformers model on a causal language modeling task")
    parser.add_argument('--data_path',
                        type=str,
                        default='xstest',
                        help='Path to the training dataset, a single data path.')
    parser.add_argument('--model',
                        type=str,
                        default='llama2-7B')

    parser.add_argument('--method',
                        type=str,
                        default='Prompt')

    parser.add_argument('--baseline',
                             type=bool,
                             default=False)

    parser.add_argument('--max_new_tokens',
                        type=int,
                        default=200)

    parser.add_argument('--ratio',
                        type=float,
                        default=2.5)

    parser.add_argument('--save_path',
                        type=str,
                        default='')

    args = parser.parse_args()

    return args

class ConstractiveDecodingModel:
    def __init__(self, model, tokenizer):
        self.model = model
        self.config = self.model.config
        self.tokenizer = tokenizer

    @torch.no_grad()
    def contra_generate(self, input_within, input_without):
        maxlen_res = 512
        ratio = -3.0
        loguru.logger.info(f"ratio: {ratio}")
        dev = input_within.device
        bsz = 1

        done = torch.zeros((bsz,), device=dev).to(torch.bool)

        inds = torch.arange(bsz).to(dev).unsqueeze(1).view(-1)
        input_within = torch.index_select(input_within, 0, inds)
        input_without = torch.index_select(input_without, 0, inds)

        def score_process(score, score_without, input_within, input_without):
            score = score[:, -1, :]
            probs = score
            tok_ids_in = torch.argmax(probs, dim=-1)
            hyp_ids = torch.arange(probs.size(0), device=dev)

            tok_ids_in = torch.where(done, self.tokenizer.pad_token_id, tok_ids_in)
            input_within = torch.cat((input_within, tok_ids_in.unsqueeze(-1)), dim=-1)
            input_without = torch.cat((input_without, tok_ids_in.unsqueeze(-1)), dim=-1)

            return input_within, input_without, tok_ids_in, hyp_ids

        for _token in range(maxlen_res):
            if done.all():
                break
            score_in_output = self.model(input_within)
            score_out_output = self.model(input_without)
            score_in = score_in_output.logits.float()
            score_out = score_out_output.logits.float()

            score_without = score_out.clone()
            score_in[:, -1, :] = score_in[:, -1, :] - ratio * (score_in[:, -1, :] - score_out[:, -1, :])
            input_within, input_without, tok_ids, hyp_ids = score_process(score_in, score_without, input_within, input_without)

            done = done | tok_ids.eq(self.tokenizer.eos_token_id)

        input_within = input_within.view(bsz, -1)
        input_without = input_without.view(bsz, -1)

        return input_within, input_without


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
        "do_sample": args.do_sample,
    "ratio": args.ratio
    }

    cd_config = {
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
                                                 torch_dtype=torch.float16)

    model = ConstractiveDecodingModel(model, tokenizer)

    model.model = model.model.eval()
    print('Model loaded!')

    selected_data = CDDataset(sampled_dataset, principle=principle, conv_adapter=conv_adapter)

 
    sampled_dataset.to_json(args.output_data_file)

    data_len = len(selected_data)

    print(f"datasets len: {data_len}")

    generated_data = []
    log_probs=[]
    for index, i in tqdm(enumerate(selected_data)):
        print(f"index:{index}", flush=True)
        principle_text = i["dialogue_text_principle"]
        no_principle_text = i["dialogue_text"]
    
        chosen_answer = i["chosen_answer"]

        inputs1 = tokenizer(principle_text, return_tensors='pt')
        ids1 = inputs1.input_ids.cuda()
        att1 = inputs1.attention_mask.cuda()
        inputs2 = tokenizer(no_principle_text, return_tensors='pt')
        ids2 = inputs2.input_ids.cuda()
        att2 = inputs2.attention_mask.cuda()

        generate_ids1= model.contra_generate(
            ids1.cuda(), ids2.cuda())

        inputs = no_principle_text
        principle = principle

        len_principal = len(ids1[0])
        contra_output = tokenizer.decode(generate_ids1[0][0][len_principal:])


        data_points = {
            "id": index,
            "inputs": inputs,
            "principal": principle,
            "cd_output": contra_output,
            "chosen_answer": chosen_answer,
        }

        generated_data.append(data_points)

        with open(args.output_result_file, 'w') as f:
            json.dump(generated_data, f, indent=4)
