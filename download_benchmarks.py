from datasets import load_dataset
import pandas as pd
import re
import os

# 确保路径
os.makedirs("datasets", exist_ok=True)

# 下载 GSM8K
gsm8k = load_dataset("openai/gsm8k", "main", split="test")
gsm8k_samples = []
for item in gsm8k.select(range(800)):
    question = item['question'].strip()
    answer = item['answer']
    # 抽取 #### 后的数字
    match = re.search(r'####\s*(-?\d+)', answer)
    if match:
        label = match.group(1)
    else:
        label = None
    gsm8k_samples.append({
        "question": question,
        "label": label
    })

# 下载 ARC-Easy
arc_easy = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
arc_easy_samples = []
for item in arc_easy.select(range(800)):
    question = item['question'].strip()
    choices_text = item['choices']['text']
    choices_label = item['choices']['label']
    choices = "\n".join([f"{label}: {text.strip()}" for label, text in zip(choices_label, choices_text)])
    label = item['answerKey'].strip()
    arc_easy_samples.append({
        "question": question,
        "choices": choices,
        "label": label
    })

# 下载 BoolQ
boolq = load_dataset("google/boolq", split="train")
boolq_samples = []
for item in boolq.select(range(800)):
    question = item['question'].strip()
    passage = item['passage'].strip()
    label = str(item['answer']).lower()  # true/false
    boolq_samples.append({
        "question": question,
        "passage": passage,
        "label": label
    })

# 将数据保存成csv
pd.DataFrame(gsm8k_samples).to_csv("datasets/gsm8k_test800.csv", index=False, encoding="utf-8")
pd.DataFrame(arc_easy_samples).to_csv("datasets/arc_easy_test800.csv", index=False, encoding="utf-8")
pd.DataFrame(boolq_samples).to_csv("datasets/boolq_train800.csv", index=False, encoding="utf-8")

print("✅ 数据集下载和处理完成，已储存到 datasets 文件夹！")