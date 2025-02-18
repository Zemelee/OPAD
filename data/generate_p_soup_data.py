import json
import os
import random

# 固定随机种子
random.seed(42)

# 指定目录路径
folder_path = './data/psoups'  # 替换为你的文件夹路径
output_samples = []
# 保存到新文件
output_file_path = os.path.join(folder_path, './data/P_soup/train_sampled_output_with_principle.json')


# 获取所有 JSON 文件的文件名
json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]

# 处理每个 JSON 文件
for json_file in json_files:
    file_path = os.path.join(folder_path, json_file)
    
    # 打开并读取 JSON 文件
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # 确保数据足够大来抽取 100 个条目
    if len(data) < 50:
        print(f"{json_file} 里的条目不足 100 个，跳过")
        continue
    
    # 随机抽取 100 个条目
    sampled_data = random.sample(data, 50)
    
    # 处理每个抽取的样本
    for sample in sampled_data:
        instruction = sample.get('instruction', '')
        
        # 提取 'principle' 和 'input' 部分
        if "Generate a response" in instruction:
            principle_split = instruction.split("Generate a response", 1)
            principle = "Generate a response" + " "+ principle_split[1].strip()
            input_part = principle_split[0].strip()
        else:
            principle = ''
            input_part = instruction.strip()
        
       # 创建新的字典条目
        formatted_sample = {
            "index": len(output_samples) + 1,
            "principle": principle,
            "input": input_part,
            "output": sample.get('output', '')
        }
        
        output_samples.append(formatted_sample)
        with open(output_file_path, 'w') as f:
            json.dump(output_samples, f, indent=2)

print(f"所有文件处理完毕，结果已保存到 {output_file_path}")