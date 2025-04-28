import openai
import pandas as pd
import os
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ===== 读取OpenAI API Key =====
from utils import config_manager
config_manager = config_manager.ConfigManager()
api_key = config_manager.get_api_key("openai")
client = openai.OpenAI(api_key=api_key)

# ===== 基础配置 =====
SAVE_DIR = "results/benchmarks"
DATASET_CSV = "datasets/arc_challenge_test800.csv"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs("datasets", exist_ok=True)

# ===== 下载arc_challenge数据集并保存 =====
def download_arc_challenge():
    if os.path.exists(DATASET_CSV):
        print("✅ 已存在arc_challenge_test800.csv，跳过下载。")
        return
    
    print("🚀 正在下载 ARC-Challenge...")
    arc_challenge = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    arc_challenge = arc_challenge.select(range(800))

    samples = []
    for item in arc_challenge:
        question = item['question'].strip()
        choices_text = item['choices']['text']
        choices_label = item['choices']['label']
        choices_formatted = ""
        for label, text in zip(choices_label, choices_text):
            choices_formatted += f"{label}: {text.strip()}\n"
        label = item['answerKey'].strip()
        samples.append({
            "question": question,
            "choices": choices_formatted.strip(),
            "label": label
        })

    df = pd.DataFrame(samples)
    df.to_csv(DATASET_CSV, index=False, encoding="utf-8")
    print(f"✅ ARC-Challenge数据集已保存到: {DATASET_CSV}")

# ===== 推理阶段 =====
def infer_and_save(model_name, folder_name):
    df = pd.read_csv(DATASET_CSV)
    result_dir = os.path.join(SAVE_DIR, folder_name)
    os.makedirs(result_dir, exist_ok=True)

    predictions = []
    for idx, row in df.iterrows():
        prompt = (
            f"Read the question and options carefully. Select the correct option (A/B/C/D).\n\n"
            f"Question: {row['question']}\n"
            f"Options:\n{row['choices']}\n\n"
            "Respond with only A, B, C, or D. like: A"
        )

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            predicted = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"❌ 推理失败 idx={idx}, 错误: {e}")
            predicted = "error"

        predictions.append(predicted)

    df["prediction"] = predictions
    save_path = os.path.join(result_dir, "arc_challenge_test800_results.csv")
    df.to_csv(save_path, index=False)
    print(f"✅ 推理完成，保存到: {save_path}")

# ===== 评估阶段 =====
def evaluate_and_save(folder_name):
    file_path = os.path.join(SAVE_DIR, folder_name, "arc_challenge_test800_results.csv")
    df = pd.read_csv(file_path)

    number_to_letter = {"1": "a", "2": "b", "3": "c", "4": "d"}

    def normalize(label):
        if isinstance(label, str):
            label = label.strip().lower().replace('"', '').replace("'", '').replace('*', '')
            return number_to_letter.get(label, label)
        return label

    df["prediction_clean"] = df["prediction"].apply(normalize)
    df["label_clean"] = df["label"].apply(normalize)

    true_labels = df["label_clean"]
    pred_labels = df["prediction_clean"]

    accuracy = accuracy_score(true_labels, pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average="macro", zero_division=0)

    print(f"\n📊 Evaluation on {folder_name}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    metrics_path = os.path.join(SAVE_DIR, folder_name, "arc_challenge_test800_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
    print(f"✅ Metrics 保存到: {metrics_path}")

# ===== 主程序 =====
def main():
    download_arc_challenge()

    model_list = [
        ("gpt-4o", "benchmark-4o"),
        ("ft:gpt-4o-2024-08-06:personal:thinking-3000:BPOh2ica", "benchmark-4o-thinking"),
        ("ft:gpt-4o-2024-08-06:personal:feeling-1600-reversed:BQsGBsUO", "benchmark-4o-feeling-reversed")
    ]

    for model_name, folder_name in model_list:
        infer_and_save(model_name, folder_name)
        evaluate_and_save(folder_name)

if __name__ == "__main__":
    main()
