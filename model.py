

import torch
import numpy as np
from utils import top_p_logits,  denoise_logits



def compute_perplexity(log_probs):
    """
    Compute perplexity given a list of log-probabilities.
    log_probs: List or array of log-probabilities for each token in the sequence.
    """
    n = len(log_probs)
    avg_log_prob = np.mean(log_probs)
    perplexity = np.exp(-avg_log_prob)
    return perplexity


class ConstractiveDecodingModel:

    def __init__(self, model, tokenizer):
        self.model = model
        self.config = self.model.config
        self.tokenizer = tokenizer

    @torch.no_grad()
    def contra_generate(self, input_within, input_without, attention_mask_in, attention_mask_out, **kwargs):
        """
        Generate response
        """
        maxlen_res = kwargs.pop('max_new_tokens', 20)
        temperature = kwargs.pop('temperature', 0.5)
        topp = kwargs.pop('topp', 0.8)
        ratio = kwargs.pop('ratio', 0)
        do_sample = kwargs.pop('do_sample', False)

        dev = input_within.device
        bsz = input_within.size(0)

        done = torch.zeros((bsz,), device=dev).to(torch.bool)

        inds = torch.arange(bsz).to(dev).unsqueeze(1).view(-1)
        input_within = torch.index_select(input_within, 0, inds)
        input_without = torch.index_select(input_without, 0, inds)

        init_length_in = input_within.size(1)
    

        def score_process(score, sys_score, input_within, input_without):
            score = score[:, -1, :]    ####相当于一阶梯度项
            sys_score = sys_score[:, -1, :]  ### the original score_in

            # nucleus sampling
            score = torch.softmax(score.div(temperature), dim=-1)
            sys_score = torch.softmax(sys_score.div(temperature), dim=-1)
            probs = score.clone()
            sys_probs = top_p_logits(sys_score, topp=topp, filter_value=0)
            sys_mask = sys_probs.ne(0)      ####the largest logit idx，相当于公式中的归一化和放大项

            probs = probs * sys_mask

            if do_sample:
                probs = denoise_logits(probs, sys_probs)

                tok_ids = torch.multinomial(probs, 1)[:, 0]
            else:
                tok_ids = torch.argmax(probs, dim=-1)

            tok_ids = torch.where(done, self.tokenizer.pad_token_id, tok_ids)

            input_within = torch.cat((input_within, tok_ids.unsqueeze(-1)), dim=-1)
            input_without = torch.cat((input_without, tok_ids.unsqueeze(-1)), dim=-1)

            max_probs = torch.gather(probs, -1, tok_ids.unsqueeze(-1)).squeeze(-1)
            return input_within, input_without, tok_ids

        past_key_values_in = None
        past_key_values_out = None
        tok_ids = None

        log_probs=[]
        for _token in range(maxlen_res):

            if done.all():
                break

            if past_key_values_in is not None and past_key_values_out is not None:

                score_in_output = self.model(tok_ids.unsqueeze(-1), use_cache=True, attention_mask=attention_mask_in,
                                             past_key_values=past_key_values_in)

                
                score_out_output = self.model(tok_ids.unsqueeze(-1), use_cache=True, attention_mask=attention_mask_out,
                                              past_key_values=past_key_values_out)
                past_key_values_in = score_in_output.past_key_values
                past_key_values_out = score_out_output.past_key_values

            else:

                score_in_output = self.model(input_within, attention_mask=attention_mask_in, use_cache=True)  ###log probabilities
                score_out_output = self.model(input_without, attention_mask=attention_mask_out, use_cache=True)
                past_key_values_in = score_in_output.past_key_values
                past_key_values_out = score_out_output.past_key_values


            score_in = score_in_output.logits.float()
            score_out = score_out_output.logits.float()

            sys_score = score_in.clone()
            score_in[:, -1, :] = score_in[:, -1, :] * (ratio + 1) - score_out[:, -1, :] * ratio  ####logits at the last token

            input_within, input_without, tok_ids = score_process(score_in, sys_score, input_within,
                                                                 input_without)
            # log_probs.append(log_prob)
            new_attention_values = torch.ones((attention_mask_in.shape[0], 1), device=dev,
                                              dtype=attention_mask_in.dtype)

            attention_mask_in = torch.cat([attention_mask_in, new_attention_values], dim=-1)
            attention_mask_out = torch.cat([attention_mask_out, new_attention_values], dim=-1)

            done = done | tok_ids.eq(self.tokenizer.eos_token_id)

        # get all finalized candidates for each sample
        input_within = input_within[:, init_length_in:]
        input_within = input_within.view(bsz, -1)


        return input_within

