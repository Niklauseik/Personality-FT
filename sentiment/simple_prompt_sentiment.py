import openai
import pandas as pd
import os
import random
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 读取 OpenAI API Key
from utils import config_manager
config_manager = config_manager.ConfigManager()
api_key = config_manager.get_api_key("openai")

# 初始化 OpenAI 客户端
client = openai.OpenAI(api_key=api_key)

# 替换为无微调的 GPT-4o-mini
base_model = "gpt-4o-mini"

# 读取测试数据集
test_file = "datasets/sentiment.csv"
df = pd.read_csv(test_file)

# 预定义四种 MBTI 风格
mbti_prompts = {
    "ENTJ": "You are an ENTJ.",
    "ISFP": "You are an ISFP.",
    "INTP": "You are an INTP.",
    "ESFJ": "You are an ESFJ."
}

# **测试存储目录**
results_dir = "results/mbti_sentiment_simple"
os.makedirs(results_dir, exist_ok=True)

# 遍历四种 MBTI 风格
for mbti_type, system_message in mbti_prompts.items():
    print(f"🔍 Testing MBTI role: {mbti_type}")

    # 结果存储路径
    mbti_results_dir = os.path.join(results_dir, mbti_type)
    os.makedirs(mbti_results_dir, exist_ok=True)

    # **抽取 50 条数据**
    sample_df = df.sample(n=1173, random_state=42)

    # 初始化预测列表
    predictions = []

    # 遍历测试数据进行预测
    for i, row in sample_df.iterrows():
        query = row["query"].strip()
        true_label = row["answer"].strip()

        try:
            # 发送 API 请求
            response = client.chat.completions.create(
                model=base_model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": f"{query}\nOnly reply with the sentiment label: Positive, Negative, or Neutral."}
                ]
            )

            # **解析模型预测的情感类别**
            predicted_label = response.choices[0].message.content.strip()

            # 确保只返回预期的标签
            if predicted_label not in ["Positive", "Negative", "Neutral"]:
                raise ValueError(f"❌ Unexpected output: {predicted_label}")

            predictions.append(predicted_label)

        except Exception as e:
            print(f"❌ Prediction failed, error: {e}")
            predictions.append("error")  # 失败时填充 "error"
    
    # **存储预测结果**
    sample_df["prediction"] = predictions
    results_file = os.path.join(mbti_results_dir, "sentiment_with_predictions.csv")
    sample_df.to_csv(results_file, index=False)
    print(f"✅ Predictions saved to: {results_file}")

    # **计算 Metrics**
    true_labels = sample_df["answer"].str.lower()
    pred_labels = sample_df["prediction"].str.lower()

    accuracy = accuracy_score(true_labels, pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average="macro", zero_division=0)

    # **保存 Metrics 结果**
    metrics_file = os.path.join(mbti_results_dir, "metrics.txt")
    with open(metrics_file, "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")

    print(f"✅ Evaluation metrics saved to: {metrics_file}")
