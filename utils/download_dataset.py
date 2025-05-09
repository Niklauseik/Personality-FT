from datasets import load_dataset
import pandas as pd

# 加载数据集
ds = load_dataset("ChanceFocus/flare-fiqasa")

# 提取 query 和 answer，并合并数据
df = pd.concat([
    pd.DataFrame(ds["test"])[["text", "answer"]],
    pd.DataFrame(ds["train"])[["text", "answer"]],
    pd.DataFrame(ds["valid"])[["text", "answer"]]
])

# 保存到 CSV 文件
csv_path = "datasets/fiqasa.csv"
df.to_csv(csv_path, index=False)

print(f"数据已保存至 {csv_path}")
