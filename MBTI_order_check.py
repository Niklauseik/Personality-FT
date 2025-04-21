import openai
import json
import os
import time
import pandas as pd
from collections import Counter
from utils import config_manager

# 读取 OpenAI API Key
config_manager = config_manager.ConfigManager()
api_key = config_manager.get_api_key("openai")

# 初始化 OpenAI 客户端
client = openai.OpenAI(api_key=api_key)

# 你的微调模型 ID
fine_tuned_model = "ft:gpt-4o-mini-2024-07-18:personal:entj-general:B9VO24mx"

# **读取 MBTI 原始数据**
test_file = "datasets/MBTI.json"
with open(test_file, "r", encoding="utf-8") as f:
    original_mbti_data = json.load(f)

# **创建 选项互换 版本的 MBTI 数据**
reversed_mbti_data = []
for item in original_mbti_data:
    reversed_mbti_data.append({
        "question": item["question"],
        "choice_a": item["choice_b"],  # 互换 A/B
        "choice_b": item["choice_a"]
    })

# **合并 原始 + 反转 数据**
combined_mbti_data = original_mbti_data + reversed_mbti_data

# **手动输入 `x` 变量，决定结果存储路径**
x = input("请输入测试名称（结果将存入 results/x 文件夹）： ").strip()

# 结果存储目录
results_dir = f"results/{x}"
os.makedirs(results_dir, exist_ok=True)

# **系统消息 - 设定测试规则**
system_message = (
    "You are taking an MBTI personality test. "
    "For each question, choose either 'a' or 'b' as the answer. "
    "Respond with only a single letter ('a' or 'b') per line. "
)

# **按批次请求 API**
BATCH_SIZE = 20  # 每批 20 个问题
MAX_RETRIES = 3  # 最多重试 3 次
predictions = []

num_batches = len(combined_mbti_data) // BATCH_SIZE + (1 if len(combined_mbti_data) % BATCH_SIZE else 0)

for i in range(num_batches):
    batch_data = combined_mbti_data[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
    
    # **构造 Prompt**
    query = "\n".join([
        f"Q{idx+1}: {q['question']} (a) {q['choice_a']['text']} OR (b) {q['choice_b']['text']}?"
        for idx, q in enumerate(batch_data)
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

            # **解析模型回答**
            batch_answers = response.choices[0].message.content.strip().split("\n")
            batch_answers = [ans.strip().lower() for ans in batch_answers]  # 转小写

            # **检查返回格式**
            if len(batch_answers) != len(batch_data):
                raise ValueError(f"❌ 期望 {len(batch_data)} 个答案，但收到 {len(batch_answers)} 个")

            # **确保所有答案都是 a 或 b**
            valid_choices = {"a", "b"}
            if any(ans not in valid_choices for ans in batch_answers):
                raise ValueError(f"❌ 发现无效答案")

            # **解析成功，存入 predictions**
            predictions.extend(batch_answers)
            success = True

        except Exception:
            retry_count += 1
            time.sleep(1)  # 等待 1 秒再重试

    if not success:
        print(f"❌ Batch {i+1}/{num_batches} 失败，已重试 {MAX_RETRIES} 次，跳过该批次")
        continue  # 跳过这个 batch，继续下一个 batch

# **统计 原始 和 反转 数据的预测结果**
original_predictions = predictions[:len(original_mbti_data)]
reversed_predictions = predictions[len(original_mbti_data):]

# **映射 a/b 到 MBTI 维度**
original_mbti_votes = {"E": 0, "I": 0, "S": 0, "N": 0, "T": 0, "F": 0, "J": 0, "P": 0}
reversed_mbti_votes = {"E": 0, "I": 0, "S": 0, "N": 0, "T": 0, "F": 0, "J": 0, "P": 0}

for (q1, choice1), (q2, choice2) in zip(
    zip(original_mbti_data, original_predictions),
    zip(reversed_mbti_data, reversed_predictions)
):
    original_value = q1["choice_a"]["value"] if choice1 == "a" else q1["choice_b"]["value"]
    reversed_value = q2["choice_a"]["value"] if choice2 == "a" else q2["choice_b"]["value"]
    original_mbti_votes[original_value] += 1
    reversed_mbti_votes[reversed_value] += 1

# **计算最终 MBTI 类型**
original_mbti_result = "".join([
    "E" if original_mbti_votes["E"] > original_mbti_votes["I"] else "I",
    "S" if original_mbti_votes["S"] > original_mbti_votes["N"] else "N",
    "T" if original_mbti_votes["T"] > original_mbti_votes["F"] else "F",
    "J" if original_mbti_votes["J"] > original_mbti_votes["P"] else "P"
])

reversed_mbti_result = "".join([
    "E" if reversed_mbti_votes["E"] > reversed_mbti_votes["I"] else "I",
    "S" if reversed_mbti_votes["S"] > reversed_mbti_votes["N"] else "N",
    "T" if reversed_mbti_votes["T"] > reversed_mbti_votes["F"] else "F",
    "J" if reversed_mbti_votes["J"] > reversed_mbti_votes["P"] else "P"
])

# **存储预测结果**
results_csv_file = os.path.join(results_dir, "mbti_predictions.csv")
df = pd.DataFrame({
    "Question": [q["question"] for q in original_mbti_data],
    "Choice A": [q["choice_a"]["text"] for q in original_mbti_data],
    "Choice B": [q["choice_b"]["text"] for q in original_mbti_data],
    "Model Choice (Original Order)": original_predictions,
    "Model Choice (Reversed Order)": reversed_predictions
})
df.to_csv(results_csv_file, index=False)

# **存储最终 MBTI 结果**
mbti_result_file = os.path.join(results_dir, "mbti_result.txt")
with open(mbti_result_file, "w") as f:
    f.write(f"Original Order MBTI: {original_mbti_result}\n")
    f.write(f"Reversed Order MBTI: {reversed_mbti_result}\n")

print(f"✅ 测试完成，结果已保存至: {results_csv_file}")
print(f"✅ MBTI 结果已保存至: {mbti_result_file}")
print(f"📊 **原始顺序 MBTI: {original_mbti_result}**")
print(f"📊 **反转顺序 MBTI: {reversed_mbti_result}**")
