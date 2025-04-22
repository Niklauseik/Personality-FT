import os
import json
import tiktoken

# 修改为你的 DPO 文件路径
DPO_FILE = "datasets/mbti_dpo/ENTJ_3200.jsonl"

# 选择 tokenizer（gpt-4 / gpt-4o / gpt-3.5 都用 cl100k_base）
tokenizer = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    """计算文本 token 数"""
    return len(tokenizer.encode(text))

# 开始统计
total_tokens = 0
sample_count = 0

with open(DPO_FILE, "r", encoding="utf-8") as f:
    for line in f:
        sample = json.loads(line)
        sample_count += 1

        prompt = sample["input"]["messages"][0]["content"]
        preferred = sample["preferred_output"][0]["content"]
        non_preferred = sample["non_preferred_output"][0]["content"]

        total_tokens += count_tokens(prompt) + count_tokens(preferred) + count_tokens(non_preferred)

# 输出结果
print(f"✅ 文件名: {os.path.basename(DPO_FILE)}")
print(f"📦 样本总数: {sample_count}")
print(f"🔢 总 token 数: {total_tokens}")
