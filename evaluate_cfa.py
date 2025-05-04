import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def clean_text(text):
    """统一格式：小写 + 去空格"""
    if not isinstance(text, str):
        return "error"
    return text.lower().strip()

def evaluate_cfa(file_path):
    df = pd.read_csv(file_path)

    df["prediction_clean"] = df["prediction"].apply(clean_text)
    df["answer_clean"] = df["answer"].apply(clean_text)

    y_true = df["answer_clean"]
    y_pred = df["prediction_clean"]

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    return accuracy, precision, recall, f1

def main():
    results_dir = "results/finben_cfa"
    all_results = []

    for file_name in os.listdir(results_dir):
        if not file_name.endswith("_results.csv"):
            continue

        file_path = os.path.join(results_dir, file_name)
        print(f"🚀 Evaluating: {file_name}")

        accuracy, precision, recall, f1 = evaluate_cfa(file_path)
        all_results.append((file_name, accuracy, precision, recall, f1))

        # 保存每个模型的 txt
        metrics_path = os.path.join(results_dir, file_name.replace("_results.csv", "_metrics.txt"))
        with open(metrics_path, "w") as f:
            f.write(f"Accuracy:  {accuracy:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall:    {recall:.4f}\n")
            f.write(f"F1-score:  {f1:.4f}\n")

        print(f"✅ Metrics saved to {metrics_path}")

    # 打印总表
    print("\n📊 Summary on CFA:")
    summary_df = pd.DataFrame(all_results, columns=["File", "Accuracy", "Precision", "Recall", "F1-score"])
    print(summary_df.to_markdown(index=False))

    # 可选：保存为 CSV
    summary_df.to_csv(os.path.join(results_dir, "evaluation_summary.csv"), index=False)

if __name__ == "__main__":
    main()
