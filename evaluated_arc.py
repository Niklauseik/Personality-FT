import pandas as pd
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 数字到字母的映射
number_to_letter = {
    "1": "a",
    "2": "b",
    "3": "c",
    "4": "d"
}

def normalize_label_arc(label):
    """统一ARC标签格式：数字转字母，小写"""
    if isinstance(label, str):
        label = label.strip().lower().replace('"', '').replace("'", '').replace('*', '')
        if label in number_to_letter:
            return number_to_letter[label]
        else:
            return label
    return label

def evaluate_arc(file_path):
    """评估单个ARC结果文件，返回四项指标"""
    df = pd.read_csv(file_path)

    df["prediction_clean"] = df["prediction"].apply(normalize_label_arc)
    df["label_clean"] = df["label"].apply(normalize_label_arc)

    true_labels = df["label_clean"]
    pred_labels = df["prediction_clean"]

    accuracy = accuracy_score(true_labels, pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average="macro", zero_division=0
    )

    return accuracy, precision, recall, f1

def main():
    model_paths = {
        "GPT-4o 原始": "results/benchmarks/benchmark-4o/arc_easy_test800_results.csv",
        "GPT-4o-T 型人格": "results/benchmarks/benchmark-4o-thinking/arc_easy_test800_results.csv",
        "GPT-4o-F 型人格": "results/benchmarks/benchmark-4o-feeling-reversed/arc_easy_test800_results.csv"
    }

    all_results = []

    for model_name, file_path in model_paths.items():
        print(f"🚀 Evaluating {model_name}...")

        if not os.path.exists(file_path):
            print(f"⚠️  File not found: {file_path}")
            continue

        accuracy, precision, recall, f1 = evaluate_arc(file_path)
        all_results.append((model_name, accuracy, precision, recall, f1))

        # 保存 arc_easy_metrics.txt
        save_dir = os.path.dirname(file_path)
        metrics_path = os.path.join(save_dir, "arc_easy_metrics.txt")
        with open(metrics_path, "w") as f:
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"F1 Score: {f1:.4f}\n")
        print(f"✅ Metrics saved to {metrics_path}")

    # 打印汇总表格
    print("\n📊 Summary on ARC Easy:")
    summary_df = pd.DataFrame(all_results, columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score"])
    print(summary_df.to_markdown(index=False))

if __name__ == "__main__":
    main()
