import openai
import pandas as pd
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 读取 OpenAI API Key
from utils import config_manager
config_manager = config_manager.ConfigManager()
api_key = config_manager.get_api_key("openai")

# 初始化 OpenAI 客户端
client = openai.OpenAI(api_key=api_key)

# 读取测试数据集
test_file = "datasets/fiqasa.csv"
df = pd.read_csv(test_file)

# **随机抽取 n 条测试数据**
df_sample = df.sample(n=1173, random_state=42).reset_index(drop=True)


def clean_prediction(pred):
    if isinstance(pred, str):
        pred = pred.strip().lower()
        pred = pred.replace('"', '').replace("'", '').replace('*', '')
        pred = pred.strip()
    return pred


def run_test(model_name, folder_name):
    print(f"\n🚀 Running test for model: {model_name} saving to: {folder_name}")

    # 构建保存路径
    results_dir = os.path.join("results", "sentiment", folder_name)
    os.makedirs(results_dir, exist_ok=True)

    predictions = []

    # 遍历测试数据进行预测
    for i, row in df_sample.iterrows():
        text = row["text"].strip()
        true_label = row["answer"].strip()

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": (
                        "You are a financial sentiment classifier. "
                        "Respond with only one word: either 'positive', 'neutral', or 'negative'."
                    )},
                    {"role": "user", "content": (
                        "Analyze the sentiment of this statement extracted from a financial news article:\n"
                        f"{text}"
                    )}
                ]
            )
            predicted_label = response.choices[0].message.content.strip()
            predictions.append(predicted_label)

        except Exception as e:
            print(f"❌ Prediction failed at {i}, error: {e}")
            predictions.append("error")

    # 保存预测结果
    df_result = df_sample.copy()
    df_result["prediction"] = predictions

    results_file = os.path.join(results_dir, "sentiment_results.csv")
    df_result.to_csv(results_file, index=False)
    print(f"✅ Predictions saved to: {results_file}")

    # 清洗 prediction 列
    df_result["prediction"] = df_result["prediction"].apply(clean_prediction)
    df_result["answer"] = df_result["answer"].apply(lambda x: x.strip().lower() if isinstance(x, str) else x)

    # 保存清洗后的覆盖文件
    df_result.to_csv(results_file, index=False)

    # 计算 Metrics
    true_labels = df_result["answer"]
    pred_labels = df_result["prediction"]

    accuracy = accuracy_score(true_labels, pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average="macro", zero_division=0)

    # 保存 Metrics 结果
    metrics_file = os.path.join(results_dir, "metrics.txt")
    with open(metrics_file, "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")

    print(f"✅ Metrics saved to: {metrics_file}")


def main():
    model_configs = [
        ("gpt-4o", "fiqa-4o"),
        ("ft:gpt-4o-2024-08-06:personal:thinking-3000:BPOh2ica", "fiqa-4o-thinking"),
        ("ft:gpt-4o-2024-08-06:personal:feeling-1600-reversed:BQsGBsUO", "fiqa-4o-feeling-reversed")
    ]

    for model_name, folder_name in model_configs:
        run_test(model_name, folder_name)


if __name__ == "__main__":
    main()
