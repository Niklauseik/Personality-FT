import pandas as pd
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def normalize_label_boolq(label):
    """统一BoolQ标签格式：全部转成小写字符串"""
    if isinstance(label, bool):
        return str(label).lower()
    if isinstance(label, str):
        return label.strip().lower().replace('"', '').replace("'", '').replace('*', '')
    return str(label).lower()

def evaluate_boolq(file_path):
    """评估单个BoolQ结果文件"""
    df = pd.read_csv(file_path)

    df["prediction_clean"] = df["prediction"].apply(normalize_label_boolq)
    df["label_clean"] = df["label"].apply(normalize_label_boolq)

    true_labels = df["label_clean"]
    pred_labels = df["prediction_clean"]

    accuracy = accuracy_score(true_labels, pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average="macro", zero_division=0
    )

    return accuracy, precision, recall, f1

def main():
    model_paths = {
        "GPT-4o 原始": "results/benchmarks/benchmark-4o/boolq_train800_results.csv",
        "GPT-4o-T 型人格": "results/benchmarks/benchmark-4o-thinking/boolq_train800_results.csv",
        "GPT-4o-F 型人格": "results/benchmarks/benchmark-4o-feeling-reversed/boolq_train800_results.csv"
    }

    all_results = []

    for model_name, file_path in model_paths.items():
        print(f"🚀 Evaluating {model_name} on BoolQ...")

        if not os.path.exists(file_path):
            print(f"⚠️  File not found: {file_path}")
            continue

        accuracy, precision, recall, f1 = evaluate_boolq(file_path)
        all_results.append((model_name, accuracy, precision, recall, f1))

        # 保存单个模型的boolq_metrics.txt
        save_dir = os.path.dirname(file_path)
        metrics_path = os.path.join(save_dir, "boolq_metrics.txt")
        with open(metrics_path, "w") as f:
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"F1 Score: {f1:.4f}\n")
        print(f"✅ Metrics saved to {metrics_path}")

    # 打印总表
    print("\n📊 Summary on BoolQ:")
    summary_df = pd.DataFrame(all_results, columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score"])
    print(summary_df.to_markdown(index=False))

if __name__ == "__main__":
    main()
