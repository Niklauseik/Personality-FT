import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os

# 指定你的 sentiment_results.csv 路径
results_file = "results/sentiment/4o-feeling-reversed/sentiment_results.csv"  # <<< 修改成你的子文件夹路径

# 读取 CSV
df = pd.read_csv(results_file)

# 定义一个清洗函数
def clean_prediction(pred):
    if isinstance(pred, str):
        pred = pred.strip().lower()
        pred = pred.replace('"', '').replace("'", '').replace('*', '')
        pred = pred.strip()
    return pred

# 清洗 prediction 列
df["prediction"] = df["prediction"].apply(clean_prediction)

# 同时标准化 answer 列（防止 answer 有大小写问题）
df["answer"] = df["answer"].apply(lambda x: x.strip().lower() if isinstance(x, str) else x)

# 保存清洗后的 CSV（直接覆盖）
df.to_csv(results_file, index=False)
print(f"✅ Cleaned and overwritten: {results_file}")

# 重新计算指标
true_labels = df["answer"]
pred_labels = df["prediction"]

accuracy = accuracy_score(true_labels, pred_labels)
precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average="macro", zero_division=0)

# 保存新的 metrics（直接覆盖）
metrics_file = os.path.join(os.path.dirname(results_file), "metrics.txt")
with open(metrics_file, "w") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n")

print(f"✅ Updated evaluation metrics saved to: {metrics_file}")
