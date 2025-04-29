import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 要评估的模型文件夹名称
model_folders = [
    "benchmark-4o",
    "benchmark-4o-thinking",
    "benchmark-4o-feeling-reversed"
]

# 数据集文件名
datasets = ["german_400", "convfinqa_300", "cfa_1000"]

# 遍历每个模型
for folder in model_folders:
    print(f"\n🚀 Calculating metrics for model: {folder}")

    metrics_lines = []

    for dataset_name in datasets:
        file_path = os.path.join("results", "finbench", folder, f"{dataset_name}_results.csv")
        
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            continue

        df = pd.read_csv(file_path)

        y_true = df["answer"]
        y_pred = df["prediction"]

        # 特殊处理 convfinqa：四舍五入
        if "convfinqa" in dataset_name:
            y_true = y_true.apply(lambda x: str(round(float(x))) if isinstance(x, (int, float, str)) and str(x).replace('.', '', 1).isdigit() else x)
            y_pred = y_pred.apply(lambda x: str(round(float(x))) if isinstance(x, (int, float, str)) and str(x).replace('.', '', 1).isdigit() else x)

        # 通用清洗
        y_true = y_true.astype(str).str.strip().str.lower()
        y_pred = y_pred.astype(str).str.strip().str.lower()

        # 计算 Accuracy
        acc = accuracy_score(y_true, y_pred)

        # 记录输出
        metrics_lines.append(f"Dataset: {dataset_name}")
        metrics_lines.append(f"Accuracy: {round(acc, 4)}")

        # 其他两个数据集计算完整指标
        if dataset_name != "convfinqa_300":
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average="macro", zero_division=0
            )
            metrics_lines.append(f"Precision: {round(precision, 4)}")
            metrics_lines.append(f"Recall: {round(recall, 4)}")
            metrics_lines.append(f"F1 Score: {round(f1, 4)}")

        metrics_lines.append("")

    # 保存为 TXT 文件
    output_file = os.path.join("results", "finbench", folder, f"{folder}_metrics.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        for line in metrics_lines:
            f.write(line + "\n")

    print(f"✅ Saved: {output_file}")
