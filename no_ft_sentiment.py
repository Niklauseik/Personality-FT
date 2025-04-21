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

# **随机抽取 20 条测试数据**
df_sample = df.sample(n=1173, random_state=42).reset_index(drop=True)

# 结果存储目录
results_dir = "results/sentiment/no_ft_model_no_prompt"
os.makedirs(results_dir, exist_ok=True)

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
                {"role": "system", "content": "You are a financial sentiment classifier. Respond with one word"},
                {"role": "user", "content": query}
            ]
        )
        
        # 解析模型预测的情感类别
        predicted_label = response.choices[0].message.content.strip()
        predictions.append(predicted_label)
    
    except Exception as e:
        print(f"❌ Prediction failed, error: {e}")
        predictions.append("error")  # 失败时填充 "error"

# **将预测结果添加到 DataFrame**
df_sample["prediction"] = predictions

# **保存预测结果 CSV**
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
