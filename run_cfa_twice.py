import os
import openai
import pandas as pd
from utils import config_manager
import time

# 获取 API Key 和初始化客户端
config_manager = config_manager.ConfigManager()
api_key = config_manager.get_api_key("openai")
client = openai.OpenAI(api_key=api_key)

# 加载 CFA 数据集
cfa_df = pd.read_csv("datasets/finben/cfa_1000.csv")

# 通用清洗函数
def clean_prediction(pred):
    if isinstance(pred, str):
        pred = pred.strip().lower()
        pred = pred.replace('"', '').replace("'", '').replace('*', '')
    return pred

# 执行模型调用 + 重试机制
def call_with_retries(prompt, model_name, max_retries=3):
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"⚠️ Attempt {attempt} failed: {e}")
            time.sleep(1.5 * attempt)  # 指数退避
    return "error"

# 单次运行函数
def run_cfa(model_name, model_tag, run_id):
    print(f"\n🚀 Running CFA test: {model_tag} (Run {run_id})")

    results_dir = os.path.join("results", "finben_cfa")
    os.makedirs(results_dir, exist_ok=True)

    predictions = []
    for i, row in cfa_df.iterrows():
        prompt = f"{row['text']}\n\nOnly respond with A, B, or C. For example: C"
        predicted = call_with_retries(prompt, model_name)
        predictions.append(predicted)

    # 保存结果
    df_result = cfa_df.copy()
    df_result["prediction"] = predictions
    df_result["prediction"] = df_result["prediction"].apply(clean_prediction)
    df_result["answer"] = df_result["answer"].astype(str).str.lower().str.strip()

    result_file = os.path.join(results_dir, f"{model_tag}_run{run_id}_results.csv")
    df_result.to_csv(result_file, index=False, encoding="utf-8")
    print(f"✅ Saved: {result_file}")

# 主函数：三个模型各跑两次
def main():
    model_configs = [
        ("gpt-4o", "4o"),
        ("ft:gpt-4o-2024-08-06:personal:thinking-3000:BPOh2ica", "4o-thinking"),
        ("ft:gpt-4o-2024-08-06:personal:feeling-1600-reversed:BQsGBsUO", "4o-feeling-reversed")
    ]

    for model_name, model_tag in model_configs:
        for run_id in [1, 2]:
            run_cfa(model_name, model_tag, run_id)

if __name__ == "__main__":
    main()
