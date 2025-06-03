from datasets import load_dataset
import pandas as pd
import os

# ✅ 加载 flare-edtsum 数据集（只取 test split）
dataset = load_dataset("ChanceFocus/flare-edtsum", split="test")

# ✅ 提取 query 和 answer 两列
df = pd.DataFrame({
    "query": dataset["query"],
    "answer": dataset["answer"]
})

# ✅ 创建目标目录并保存
os.makedirs("datasets/finben", exist_ok=True)
df.to_csv("datasets/finben/edtsum_test.csv", index=False, encoding="utf-8")

print("✅ 已保存到 datasets/finben/edtsum_test.csv")