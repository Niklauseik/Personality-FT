import os
import json
import tiktoken  # OpenAI 的 tokenizer

# 设置文件路径
FILE_PATH = "datasets/mbti_ft/ISFP_general.jsonl"
tokenizer = tiktoken.get_encoding("cl100k_base")

def count_tokens(text):
    return len(tokenizer.encode(text))

total_tokens = 0
num_entries = 0

with open(FILE_PATH, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        messages = item.get("messages", [])
        for msg in messages:
            total_tokens += count_tokens(msg.get("content", ""))
        num_entries += 1

print(f"✅ 文件名: {os.path.basename(FILE_PATH)}")
print(f"📦 总样本数: {num_entries}")
print(f"🔢 总 token 数: {total_tokens}")
