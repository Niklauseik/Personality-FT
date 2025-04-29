from datasets import load_dataset
import pandas as pd
import os

# 创建目标目录
os.makedirs("datasets/finben", exist_ok=True)

# 1. flare-german
german = load_dataset("TheFinAI/flare-german", split="train[:400]")
german_df = pd.DataFrame({
    "text": german["query"],
    "answer": german["answer"]
})
german_df.to_csv("datasets/finben/german_400.csv", index=False)

# 2. flare-convfinqa
convfinqa = load_dataset("ChanceFocus/flare-convfinqa", split="test[:300]")
convfinqa_df = pd.DataFrame({
    "text": convfinqa["query"],
    "answer": convfinqa["answer"]
})
convfinqa_df.to_csv("datasets/finben/convfinqa_300.csv", index=False)

# 3. flare-cfa
cfa = load_dataset("TheFinAI/flare-cfa", split="test[:1000]")
cfa_df = pd.DataFrame({
    "text": cfa["text"],
    "answer": cfa["answer"]
})
cfa_df.to_csv("datasets/finben/cfa_1000.csv", index=False)

print("✅ 所有数据已保存到 datasets/finben/")
