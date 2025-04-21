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

# 替换为你的微调模型 ID
fine_tuned_model = "gpt-4o-mini"

# 读取测试数据集
test_file = "datasets/sentiment.csv"
df = pd.read_csv(test_file)

# **随机抽取 1173 条测试数据**
df_sample = df.sample(n=1173, random_state=42).reset_index(drop=True)

# 定义不同的 Prompt 版本
prompts = {
    "prompt_1_basic": "You are a financial sentiment classifier. Respond with one word.",
    "prompt_2_expert": "You are an expert financial sentiment classifier. Classify the following financial post as Positive, Negative, or Neutral. Respond with only one word.",
    "prompt_3_examples": """You are an expert in financial sentiment analysis. Given a financial post, classify its sentiment as Positive, Negative, or Neutral. 
Example:
- "The stock price surged after the earnings report." → Positive
- "The company reported massive losses this quarter." → Negative
- "Markets remained flat throughout the trading day." → Neutral
Now classify the following:""",
    "prompt_4_format": """You are a financial sentiment classifier. Classify the given financial post into one of the following categories: 
- Positive
- Negative
- Neutral

Respond strictly with one of the three words above. Do not provide any explanation.""",
    "prompt_5_confidence": """You are a financial sentiment classifier. Analyze the sentiment of the given financial post and classify it as Positive, Negative, or Neutral. 
Additionally, provide a confidence score (High, Medium, Low) based on the clarity of the sentiment in the text. 
Format your response as: "Sentiment: [Positive/Negative/Neutral], Confidence: [High/Medium/Low]" """
}

# 结果存储目录
results_base_dir = "results/sentiment/"
os.makedirs(results_base_dir, exist_ok=True)

# 遍历不同的 Prompt 进行测试
for prompt_name, system_prompt in prompts.items():
    print(f"🚀 Running test for: {prompt_name}")

    # 初始化预测列表
    predictions = []

    # 遍历测试数据进行预测
    for i, row in df_sample.iterrows():
        query = row["query"].strip()
        true_label = row["answer"].strip()

        try:
            # 发送 API 请求
            response = client.chat.completions.create(
                model=fine_tuned_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ]
            )

            # 解析模型预测的情感类别
            predicted_label = response.choices[0].message.content.strip()

            # 处理 Confidence Prompt 的特殊格式
            if "Confidence:" in predicted_label:
                predicted_label = predicted_label.split(",")[0].replace("Sentiment:", "").strip()

            predictions.append(predicted_label)

        except Exception as e:
            print(f"❌ Prediction failed, error: {e}")
            predictions.append("error")  # 失败时填充 "error"

    # **将预测结果添加到 DataFrame**
    df_sample["prediction"] = predictions

    # **保存预测结果 CSV**
    results_dir = os.path.join(results_base_dir, prompt_name)
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, "sentiment_with_predictions.csv")
    df_sample.to_csv(results_file, index=False)
    print(f"✅ Predictions saved to: {results_file}")

    # **计算 Metrics**
    true_labels = df_sample["answer"].str.lower()
    pred_labels = df_sample["prediction"].str.lower()

    accuracy = accuracy_score(true_labels, pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average="macro", zero_division=0)

    # **保存 Metrics 结果**
    metrics_file = os.path.join(results_dir, "metrics.txt")
    with open(metrics_file, "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")

    print(f"✅ Evaluation metrics saved to: {metrics_file}")

print("🎉 All tests completed!")
