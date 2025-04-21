import os
import json
import tiktoken  # OpenAI 的 tokenizer

# 设置数据集路径
DATASET_DIR = "datasets/mbti_ft"

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

        total_tokens = sum(
            count_tokens(item.get("instruction", "")) + count_tokens(item.get("output", ""))
            for item in data
        )

        results[file_name] = total_tokens

# 输出统计结果
for file, tokens in results.items():
    print(f"{file}: {tokens} tokens")
