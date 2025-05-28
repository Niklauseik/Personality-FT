from datasets import load_dataset
import pandas as pd
import os

# 创建目标目录
os.makedirs("datasets/finben", exist_ok=True)

# 1. flare-german
german = load_dataset("TheFinAI/flare-german", split="train[:700]")
german_df = pd.DataFrame({
    "text": german["query"],
    "answer": german["answer"]
})
german_df.to_csv("datasets/finben/german_700.csv", index=False)

# 2. flare-cfa
cfa = load_dataset("TheFinAI/flare-cfa", split="test[:1000]")
cfa_df = pd.DataFrame({
    "text": cfa["text"],
    "answer": cfa["answer"]
})
cfa_df.to_csv("datasets/finben/cfa_1000.csv", index=False)

# 3. flare-sm-bigdata
bigdata = load_dataset("TheFinAI/flare-sm-bigdata", split="test")
bigdata_df = pd.DataFrame({
    "text": bigdata["query"],
    "answer": bigdata["answer"]
})
bigdata_df.to_csv("datasets/finben/bigdata_1400.csv", index=False)

# 4. flare-headlines（只取前 2000 条）
headlines = load_dataset("ChanceFocus/flare-headlines", split="test[:2000]")
headlines_df = pd.DataFrame({
    "text": headlines["query"],
    "answer": headlines["answer"]
})
headlines_df.to_csv("datasets/finben/headlines_2000.csv", index=False)

print("✅ 所有数据已保存到 datasets/finben/")
