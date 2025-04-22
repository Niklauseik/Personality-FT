import openai
import pandas as pd
import os
import math
import time

# 读取 OpenAI API Key
from utils import config_manager
config_manager = config_manager.ConfigManager()
api_key = config_manager.get_api_key("openai")

# 初始化 OpenAI 客户端
client = openai.OpenAI(api_key=api_key)

# 替换为你的微调模型 ID
fine_tuned_model = "ft:gpt-4o-mini-2024-07-18:personal:entj-general:B9VO24mx"

# 读取 MBTI 问题数据集（完整数据）
test_file = "datasets/mbti_questions.csv"
df = pd.read_csv(test_file)

# **手动输入 `x` 变量，决定结果存储路径**
x = input("请输入测试名称（结果将存入 results/x 文件夹）： ").strip()

# 结果存储目录
results_dir = f"results/{x}"
os.makedirs(results_dir, exist_ok=True)

# **映射维度到 MBTI 选项**
dimension_map = {0: "(E or I)", 1: "(S or N)", 2: "(T or F)", 3: "(J or P)"}

# **构造 API 请求的 Prompt**
system_message = (
    "You are answering an MBTI personality test."
    "For each question, choose the personality trait that best fits your response. "
    "Respond with only one letter per line: \n"
    "E or I (Extraversion vs. Introversion)\n"
    "S or N (Sensing vs. Intuition)\n"
    "T or F (Thinking vs. Feeling)\n"
    "J or P (Judging vs. Perceiving)\n"
    "Your response must only contain a single letter (E, I, S, N, T, F, J, or P) on each line."
)

# **按批次请求 API**
BATCH_SIZE = 20  # 每批 20 个问题
MAX_RETRIES = 3  # 最多重试 3 次
predictions = []

num_batches = math.ceil(len(df) / BATCH_SIZE)

for i in range(num_batches):
    batch_df = df.iloc[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
    
    # **在问题后面加上对应维度的选项**
    query = "\n".join([
        f"Q{idx+1}: {q} {dimension_map[d]}" 
        for idx, (q, d) in enumerate(zip(batch_df["Question"], batch_df["dimension"]))
    ])

    retry_count = 0
    success = False

    while retry_count < MAX_RETRIES and not success:
        try:
            response = client.chat.completions.create(
                model=fine_tuned_model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": query}
                ]
            )

            # **打印 API 响应，便于调试**
            model_response = response.choices[0].message.content.strip()
            print(f"📌 Batch {i+1}/{num_batches}（尝试 {retry_count + 1} 次）原始模型回答:\n{model_response}\n")

            # **解析模型回答**
            batch_answers = model_response.split("\n")

            # **检查返回格式**
            if len(batch_answers) != len(batch_df):
                raise ValueError(f"❌ 期望 {len(batch_df)} 个答案，但收到 {len(batch_answers)} 个: {batch_answers}")

            # **确保所有答案都是 E/I, S/N, T/F, J/P**
            valid_choices = {"E", "I", "S", "N", "T", "F", "J", "P"}
            if any(ans.strip() not in valid_choices for ans in batch_answers):
                raise ValueError(f"❌ 发现无效答案: {batch_answers}")

            # **解析成功，存入 predictions**
            predictions.extend([ans.strip() for ans in batch_answers])
            success = True

        except Exception as e:
            print(f"⚠️ API 调用失败（尝试 {retry_count + 1}/{MAX_RETRIES}），错误: {e}")
            retry_count += 1
            time.sleep(1)  # 等待 1 秒再重试

    if not success:
        print(f"❌ Batch {i+1}/{num_batches} 失败，已重试 {MAX_RETRIES} 次，跳过该批次")
        continue  # 跳过这个 batch，继续下一个 batch

# **存储模型的选择**
df["Prediction"] = predictions
results_csv_file = os.path.join(results_dir, "mbti_predictions.csv")
df.to_csv(results_csv_file, index=False)

# **计算最终 MBTI 类型**
mbti_result = ""
mbti_result += "E" if list(predictions).count("E") > list(predictions).count("I") else "I"
mbti_result += "S" if list(predictions).count("S") > list(predictions).count("N") else "N"
mbti_result += "T" if list(predictions).count("T") > list(predictions).count("F") else "F"
mbti_result += "J" if list(predictions).count("J") > list(predictions).count("P") else "P"

# **存储最终 MBTI 结果**
mbti_result_file = os.path.join(results_dir, "mbti_result.txt")
with open(mbti_result_file, "w") as f:
    f.write(f"Model MBTI Type: {mbti_result}\n")

print(f"✅ 测试完成，结果已保存至: {results_csv_file}")
print(f"✅ MBTI 结果已保存至: {mbti_result_file}")
print(f"📊 **最终 MBTI 类型: {mbti_result}**")
