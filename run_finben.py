import os
import openai
import pandas as pd
from utils import config_manager
import time

# 获取 API Key 和初始化客户端
config_manager = config_manager.ConfigManager()
api_key = config_manager.get_api_key("openai")
client = openai.OpenAI(api_key=api_key)

# 数据集路径
finbench_datasets = {
    "german_400": pd.read_csv("datasets/finben/german_400.csv"),
    "convfinqa_300": pd.read_csv("datasets/finben/convfinqa_300.csv"),
    "cfa_1000": pd.read_csv("datasets/finben/cfa_1000.csv")
}

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

# 测试函数
def run_finbench(model_name, folder_name):
    print(f"\n🚀 Running FINBENCH test for model: {model_name} → folder: {folder_name}")

    results_dir = os.path.join("results", "finbench", folder_name)
    os.makedirs(results_dir, exist_ok=True)

    for dataset_name, df in finbench_datasets.items():
        print(f"\n🔹 Testing on {dataset_name}...")

        predictions = []
        for i, row in df.iterrows():
            # 构造Prompt
            if "german" in dataset_name:
                prompt = (
                    f"{row['text']}\n\n"
                    "Only respond with good or bad. For example: good"
                )
            elif "convfinqa" in dataset_name:
                prompt = (
                    f"{row['text']}\n\n"
                    "Only respond with a number. For example: 60.94"
                )
            elif "cfa" in dataset_name:
                prompt = (
                    f"{row['text']}\n\n"
                    "Only respond with A, B, or C. For example: C"
                )
            else:
                prompt = row["text"]

            predicted = call_with_retries(prompt, model_name)
            predictions.append(predicted)

        # 保存结果
        df_result = df.copy()
        df_result["prediction"] = predictions
        df_result["prediction"] = df_result["prediction"].apply(clean_prediction)
        df_result["answer"] = df_result["answer"].astype(str).str.lower().str.strip()

        result_file = os.path.join(results_dir, f"{dataset_name}_results.csv")
        df_result.to_csv(result_file, index=False, encoding="utf-8")
        print(f"✅ Saved: {result_file}")

# 主入口
def main():
    model_configs = [
        ("gpt-4o", "benchmark-4o"),
        ("ft:gpt-4o-2024-08-06:personal:thinking-3000:BPOh2ica", "benchmark-4o-thinking"),
        ("ft:gpt-4o-2024-08-06:personal:feeling-1600-reversed:BQsGBsUO", "benchmark-4o-feeling-reversed")
    ]

    for model_name, folder_name in model_configs:
        run_finbench(model_name, folder_name)

if __name__ == "__main__":
    main()
