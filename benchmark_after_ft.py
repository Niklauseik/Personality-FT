import openai
import pandas as pd
import os

# 读取 OpenAI API Key
from utils import config_manager
config_manager = config_manager.ConfigManager()
api_key = config_manager.get_api_key("openai")

# 初始化 OpenAI 客户端
client = openai.OpenAI(api_key=api_key)

# 读取测试数据集
datasets = {
    "gsm8k_test800": pd.read_csv("datasets/gsm8k_test800.csv"),
    "arc_easy_test800": pd.read_csv("datasets/arc_easy_test800.csv"),
    "boolq_train800": pd.read_csv("datasets/boolq_train800.csv")
}

# 清洗预测输出
def clean_prediction(pred):
    if isinstance(pred, str):
        pred = pred.strip().lower()
        pred = pred.replace('"', '').replace("'", '').replace('*', '')
    return pred

# 针对GSM8K特有的清洗：去掉数字里的逗号
def clean_number_text(text):
    if isinstance(text, str):
        text = text.strip().lower()
        text = text.replace(',', '')
        text = text.replace('"', '').replace("'", '').replace('*', '')
    return text

# 执行推理测试
def run_test(model_name, folder_name, skip_gsm8k_for_this_model=False):
    print(f"\n🚀 Running benchmark test for model: {model_name} saving to: {folder_name}")

    results_dir = os.path.join("results", "benchmarks", folder_name)
    os.makedirs(results_dir, exist_ok=True)

    for dataset_name, df in datasets.items():
        # 特殊规则：如果当前模型需要跳过gsm8k，且当前数据集是gsm8k，则跳过
        if skip_gsm8k_for_this_model and "gsm8k" in dataset_name:
            print(f"⏭️ Skipping {dataset_name} for {model_name} (already done)")
            continue

        print(f"\n🔹 Testing on {dataset_name}...")

        predictions = []

        for i, row in df.iterrows():
            try:
                # 根据数据集生成Prompt
                if "gsm8k" in dataset_name:
                    prompt = (
                        f"Solve the following math problem carefully and give only the final answer:\n\n"
                        f"{row['question']}\n\n"
                        "Only output the final number answer. like: 8"
                    )
                elif "arc_easy" in dataset_name:
                    prompt = (
                        f"Read the question and options carefully. Select the correct option (A/B/C/D).\n\n"
                        f"Question: {row['question']}\n"
                        f"Options:\n{row['choices']}\n\n"
                        "Respond with only A, B, C, or D. like: A"
                    )
                elif "boolq" in dataset_name:
                    prompt = (
                        f"Based on the following passage, answer whether the question is true or false.\n\n"
                        f"Passage: {row['passage']}\n\n"
                        f"Question: {row['question']}\n\n"
                        "Respond with only 'true' or 'false'. like: true"
                    )
                else:
                    prompt = row['question']

                # 调用OpenAI模型推理
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}]
                )
                predicted = response.choices[0].message.content.strip()
                predictions.append(predicted)

            except Exception as e:
                print(f"❌ Prediction failed at {i}, error: {e}")
                predictions.append("error")

        # 保存预测结果
        df_result = df.copy()
        df_result["prediction"] = predictions

        # 简单清洗 prediction
        if "gsm8k" in dataset_name:
            df_result["prediction"] = df_result["prediction"].apply(clean_number_text)
            df_result["label"] = df_result["label"].apply(clean_number_text)
        else:
            df_result["prediction"] = df_result["prediction"].apply(clean_prediction)
            df_result["label"] = df_result["label"].astype(str).str.lower().str.strip()

        dataset_result_file = os.path.join(results_dir, f"{dataset_name}_results.csv")
        df_result.to_csv(dataset_result_file, index=False, encoding="utf-8")
        print(f"✅ Results saved to: {dataset_result_file}")

# 主程序
def main():
    model_configs = [
        ("gpt-4o", "benchmark-4o", True),  # ✅ gpt-4o 只跳过 gsm8k
        ("ft:gpt-4o-2024-08-06:personal:thinking-3000:BPOh2ica", "benchmark-4o-thinking", False),
        ("ft:gpt-4o-2024-08-06:personal:feeling-1600-reversed:BQsGBsUO", "benchmark-4o-feeling-reversed", False)
    ]

    for model_name, folder_name, skip_gsm8k_for_this_model in model_configs:
        run_test(model_name, folder_name, skip_gsm8k_for_this_model=skip_gsm8k_for_this_model)

if __name__ == "__main__":
    main()
