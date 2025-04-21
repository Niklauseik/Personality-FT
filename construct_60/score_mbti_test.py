import openai
import pandas as pd
import os
import random

# 读取 OpenAI API Key
from utils import config_manager
config_manager = config_manager.ConfigManager()
api_key = config_manager.get_api_key("openai")

# 初始化 OpenAI 客户端
client = openai.OpenAI(api_key=api_key)

# 替换为你的微调模型 ID
fine_tuned_model = "ft:gpt-4o-mini-2024-07-18:personal:entj-general:B9VO24mx"

# **测试存储目录**
results_dir = "results/test_run"
os.makedirs(results_dir, exist_ok=True)

# **读取 MBTI 问题数据集**
test_file = "datasets/mbti_questions.csv"
df = pd.read_csv(test_file)

# **随机抽取 10 个问题**
sample_df = df.sample(n=10, random_state=22)

# **构造 API 请求的 Prompt**
system_message = (
    "You are answering an MBTI personality test. "
    "Respond to each question with a score from 0 to 7, where 0 means 'Strongly Disagree' and 7 means 'Strongly Agree'. "
    "Each answer must be on a new line and must contain only the number (0-7), nothing else."
)

query = "\n".join([f"Q{i+1}: {q}" for i, q in enumerate(sample_df["Question"].tolist())])

# **发送 API 请求**
try:
    response = client.chat.completions.create(
        model=fine_tuned_model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": query}
        ]
    )

    # **打印原始 API 响应**
    model_response = response.choices[0].message.content.strip()
    print(f"📌 原始模型回答:\n{model_response}\n")

    # **解析模型回答**
    model_answers = model_response.split("\n")
    
    # **检查返回格式**
    if len(model_answers) != 10:
        raise ValueError(f"❌ 期望 10 个答案，但收到 {len(model_answers)} 个: {model_answers}")

    # **尝试转换为整数**
    scores = []
    for ans in model_answers:
        try:
            score = int(ans.strip())
            if 0 <= score <= 7:
                scores.append(score)
            else:
                raise ValueError(f"❌ 非法得分: {score}（应在 0-7 之间）")
        except ValueError:
            raise ValueError(f"❌ 解析失败，非整数: {ans}")

except Exception as e:
    print(f"❌ API 调用失败，错误: {e}")
    exit(1)

# **调整得分**
sample_df["Score"] = scores
sample_df["Adjusted Score"] = sample_df.apply(
    lambda row: row["Score"] if row["polarity"] == 1 else (7 - row["Score"]), axis=1
)

# **计算 MBTI 维度得分**
mbti_scores = {0: 0, 1: 0, 2: 0, 3: 0}  # 维度分数
mbti_counts = {0: 0, 1: 0, 2: 0, 3: 0}  # 维度题目计数

for _, row in sample_df.iterrows():
    dim = row["dimension"]
    mbti_scores[dim] += row["Adjusted Score"]
    mbti_counts[dim] += 1

# **计算平均得分**
for dim in mbti_scores:
    if mbti_counts[dim] > 0:
        mbti_scores[dim] /= mbti_counts[dim]

# **确定最终 MBTI 结果**
mbti_result = ""
mbti_result += "E" if mbti_scores[0] >= 3.5 else "I"
mbti_result += "S" if mbti_scores[1] >= 3.5 else "N"
mbti_result += "T" if mbti_scores[2] >= 3.5 else "F"
mbti_result += "J" if mbti_scores[3] >= 3.5 else "P"

# **存储评分结果**
results_csv_file = os.path.join(results_dir, "mbti_scores.csv")
sample_df.to_csv(results_csv_file, index=False)

# **存储最终 MBTI 结果**
mbti_result_file = os.path.join(results_dir, "mbti_result.txt")
with open(mbti_result_file, "w") as f:
    f.write(f"Model MBTI Type: {mbti_result}\n")

print(f"✅ 测试完成，评分已保存至: {results_csv_file}")
print(f"✅ MBTI 结果已保存至: {mbti_result_file}")
