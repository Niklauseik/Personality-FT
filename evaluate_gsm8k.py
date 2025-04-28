import pandas as pd
import os
import re
from sklearn.metrics import accuracy_score

def extract_number(text):
    """从文本中提取第一个出现的整数，如果没有数字返回 'error'"""
    if not isinstance(text, str):
        return "error"
    text = text.replace(',', '')  # 去掉逗号
    match = re.search(r'\d+', text)
    if match:
        return match.group(0)
    return "error"

def evaluate_gsm8k(file_path):
    df = pd.read_csv(file_path)

    df["prediction_clean"] = df["prediction"].apply(extract_number)
    df["label_clean"] = df["label"].apply(lambda x: str(x).replace(',', '').strip() if isinstance(x, str) else str(x))

    true_labels = df["label_clean"]
    pred_labels = df["prediction_clean"]

    # 无论 prediction 有没有提取到数字，都参与计算
    accuracy = accuracy_score(true_labels, pred_labels)

    return accuracy

def main():
    model_paths = {
        "GPT-4o 原始": "results/benchmarks/benchmark-4o/gsm8k_test800_results.csv",
        "GPT-4o-T 型人格": "results/benchmarks/benchmark-4o-thinking/gsm8k_test800_results.csv",
        "GPT-4o-F 型人格": "results/benchmarks/benchmark-4o-feeling-reversed/gsm8k_test800_results.csv"
    }

    all_results = []

    for model_name, file_path in model_paths.items():
        print(f"🚀 Evaluating {model_name} on GSM8K...")

        if not os.path.exists(file_path):
            print(f"⚠️ File not found: {file_path}")
            continue

        accuracy = evaluate_gsm8k(file_path)
        all_results.append((model_name, accuracy))

        save_dir = os.path.dirname(file_path)
        metrics_path = os.path.join(save_dir, "gsm8k_metrics.txt")
        with open(metrics_path, "w") as f:
            f.write(f"Accuracy: {accuracy:.4f}\n")
        print(f"✅ Accuracy saved to {metrics_path}")

    # 打印总表
    print("\n📊 Summary on GSM8K:")
    summary_df = pd.DataFrame(all_results, columns=["Model", "Accuracy"])
    print(summary_df.to_markdown(index=False))

if __name__ == "__main__":
    main()
