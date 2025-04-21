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
fine_tuned_model = "gpt-4o-mini"

# 读取 MBTI 问题数据集（完整数据）
test_file = "datasets/mbti_questions.csv"
df = pd.read_csv(test_file)

# **手动输入 `x` 变量，决定结果存储路径**
x = input("请输入测试名称（结果将存入 results/x 文件夹）： ").strip()

# 结果存储目录
results_dir = f"results/{x}"
os.makedirs(results_dir, exist_ok=True)

# **构造 API 请求的 Prompt**
system_message = (
    "You are answering an MBTI personality test."
    "Respond to each question with a score from 0 to 6, where: \n"
    "0 = Strongly Agree\n"
    "1 = Agree\n"
    "2 = Slightly Agree\n"
    "3 = Neutral\n"
    "4 = Slightly Disagree\n"
    "5 = Disagree\n"
    "6 = Strongly Disagree\n"
    "Each answer must be on a new line and must contain only the number (0-6), nothing else."
)

# **按批次请求 API**
BATCH_SIZE = 20  # 每批 10 个问题
MAX_RETRIES = 3  # 最多重试 3 次
scores = []

num_batches = math.ceil(len(df) / BATCH_SIZE)

for i in range(num_batches):
    batch_df = df.iloc[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
    query = "\n".join([f"Q{idx+1}: {q}" for idx, q in enumerate(batch_df["Question"].tolist())])

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

            # **转换为整数**
            batch_scores = []
            for ans in batch_answers:
                try:
                    score = int(ans.strip())
                    if 0 <= score <= 6:  # ✅ 修正为 0-6
                        batch_scores.append(score)
                    else:
                        raise ValueError(f"❌ 非法得分: {score}（应在 0-6 之间）")
                except ValueError:
                    raise ValueError(f"❌ 解析失败，非整数: {ans}")

            # **解析成功，存入 scores**
            scores.extend(batch_scores)
            success = True

        except Exception as e:
            print(f"⚠️ API 调用失败（尝试 {retry_count + 1}/{MAX_RETRIES}），错误: {e}")
            retry_count += 1
            time.sleep(1)  # 等待 1 秒再重试

    if not success:
        print(f"❌ Batch {i+1}/{num_batches} 失败，已重试 {MAX_RETRIES} 次，跳过该批次")
        continue  # 跳过这个 batch，继续下一个 batch

# **存储原始评分结果**
df["Score"] = scores
results_csv_file = os.path.join(results_dir, "mbti_scores.csv")
df.to_csv(results_csv_file, index=False)

print(f"✅ 测试完成，评分已保存至: {results_csv_file}")
