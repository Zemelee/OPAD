model, F, no_principle_text, principle_text, tokenizer, torch = ""

"""
    这是关于OPAD方法的伪代码文件，不可运行
"""
# 编码成 input_ids 和 attention_mask
inputs_with_principle = tokenizer(principle_text, return_tensors="pt")
input_ids_p = inputs_with_principle["input_ids"]
attention_mask_p = inputs_with_principle["attention_mask"]
len_prefix_p = input_ids_p.shape[1]  # 原始 prompt 长度

inputs_no_principle = tokenizer(no_principle_text, return_tensors="pt")
input_ids_np = inputs_no_principle["input_ids"]
attention_mask_np = inputs_no_principle["attention_mask"]

# 初始化 past key values（缓存 attention 计算）
past_np, past_p = None, None

# 初始化生成序列
generated_ids = []
done = False
max_tokens = 512
eos_token_id = tokenizer.eos_token_id
beta = 1.0  # 可调参数，用于控制 reward 强度
temperature = 1.0  # softmax 温度参数

# === Step 1: 逐 token 生成，执行 OPAD ===
for step in range(max_tokens):
    if done:
        break
    # -- Step 1.1: forward 原始模型（带原则 prompt） --
    outputs_p = model(
        input_ids=input_ids_p if step == 0 else next_token_p.unsqueeze(-1),
        attention_mask=attention_mask_p,
        past_key_values=past_p,
    )
    logits_p = outputs_p.logits[:, -1, :]  # shape: [batch_size, vocab_size]
    past_p = outputs_p.past_key_values
    log_probs_p = F.log_softmax(logits_p / temperature, dim=-1)  # 带原则log prob
    # -- Step 1.2: forward 原始模型（无原则 prompt） --
    outputs_np = model(
        input_ids=input_ids_np if step == 0 else next_token_p.unsqueeze(-1),
        attention_mask=attention_mask_np,
        past_key_values=past_np,
    )
    logits_np = outputs_np.logits[:, -1, :]
    past_np = outputs_np.past_key_values
    log_probs_np = F.log_softmax(logits_np / temperature, dim=-1)  # 无原则log prob

    # -- Step 1.3: 计算 reward（对应公式④） --
    # r = log_probs_p - log_probs_np
    reward = log_probs_p - log_probs_np  # shape: [batch_size, vocab_size]

    # -- Step 1.4: 根据公式⑥加权重构策略分布 --
    # p(y_t) ∝ π(y_t | x, c) * exp(1/β * r)
    weighted_logits = log_probs_p + reward / beta
    probs = F.softmax(weighted_logits, dim=-1)  # shape: [batch_size, vocab_size]

    # -- Step 1.5: 采样或取最大概率词 --
    next_token_p = torch.argmax(probs, dim=-1)
    generated_ids.append(next_token_p.item())

    # -- Step 1.6: 更新输入和 attention mask --
    input_ids_p = torch.cat([input_ids_p, next_token_p.unsqueeze(-1)], dim=-1)
    input_ids_np = torch.cat([input_ids_np, next_token_p.unsqueeze(-1)], dim=-1)

    attention_mask_p = torch.cat([attention_mask_p, torch.ones((1, 1))], dim=-1)
    attention_mask_np = torch.cat([attention_mask_np, torch.ones((1, 1))], dim=-1)

    # -- Step 1.7: 检查是否生成结束 --
    if next_token_p.item() == eos_token_id:
        done = True

# === Step 2: 解码生成文本 ===
output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
