import os
import json
import random
import tiktoken  # OpenAI 的 tokenizer

# 设置数据集路径
DATASET_DIR = "datasets/mbti_raw"
OUTPUT_DIR = "datasets/mbti_samples"
SAMPLE_SIZE = 800  # 设定抽样大小

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 选择 tokenizer (GPT-4 使用 "cl100k_base")
tokenizer = tiktoken.get_encoding("cl100k_base")

def count_tokens(text):
    """计算文本的 token 数量"""
    return len(tokenizer.encode(text))

results = {}

for file_name in os.listdir(DATASET_DIR):
    if file_name.endswith(".json"):
        file_path = os.path.join(DATASET_DIR, file_name)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 抽取最多 SAMPLE_SIZE 条数据
        sampled_data = random.sample(data, min(SAMPLE_SIZE, len(data)))

        # 计算 token 数
        total_tokens = sum(
            count_tokens(item.get("instruction", "")) + count_tokens(item.get("output", ""))
            for item in sampled_data
        )
        
        results[file_name] = total_tokens

        # **保存抽样数据到 `mbti_samples` 目录**
        output_path = os.path.join(OUTPUT_DIR, file_name)
        with open(output_path, "w", encoding="utf-8") as out_f:
            json.dump(sampled_data, out_f, indent=4, ensure_ascii=False)

# 输出统计结果
for file, tokens in results.items():
    print(f"{file}: {tokens} tokens (sampled {SAMPLE_SIZE} entries)")
print(f"✅ 抽样数据已保存到 `{OUTPUT_DIR}` 目录。")
