import pandas as pd
import json
import openai
import collections

# 读取 OpenAI API Key
from utils import config_manager
config_manager = config_manager.ConfigManager()
api_key = config_manager.get_api_key("openai")

# 初始化 OpenAI 客户端
client = openai.OpenAI(api_key=api_key)

# MBTI维度映射
dimension_map = {
    0: ("E", "I"),
    1: ("S", "N"),
    2: ("T", "F"),
    3: ("J", "P")
}

# 加载问题数据
df = pd.read_csv("datasets/mbti_questions_with_dimensions.csv")

# 最终结果
final_data = []

# 对每个问题运行三次，并打印输出
for idx, row in df.iterrows():
    question = row["Question"]
    dimension = int(row["dimension"])
    dim_a, dim_b = dimension_map[dimension]

    prompt = (
        f"The following is a question from a personality test:\n\n"
        f"Q: {question}\n"
        f"If someone answers 'Yes' to this question, does it reflect more of '{dim_a}' or '{dim_b}'?\n"
        f"Only respond with one letter: '{dim_a}' or '{dim_b}'."
    )

    print(f"\n🧠 处理问题 {idx+1}: {question}")
    vote_counter = collections.Counter()

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[{"role": "user", "content": prompt}]
            )
            result = response.choices[0].message.content.strip().upper()
            print(f"  🎯 第 {attempt+1} 次回答: {result}")
            if result in (dim_a, dim_b):
                vote_counter[result] += 1
            else:
                print(f"  ⚠️ 无效回答: {result}")
        except Exception as e:
            print(f"  ❌ API 出错: {e}")

    if not vote_counter:
        print(f"❌ 无有效回答，跳过该题")
        continue

    # 多数投票结果
    yes_value = vote_counter.most_common(1)[0][0]
    no_value = dim_b if yes_value == dim_a else dim_a

    print(f"✅ 最终采用：Yes → {yes_value}, No → {no_value}")

    # 构建双版本
    final_data.append({
        "question": question,
        "choice_a": {"text": "Yes", "value": yes_value},
        "choice_b": {"text": "No", "value": no_value}
    })
    final_data.append({
        "question": question,
        "choice_a": {"text": "No", "value": no_value},
        "choice_b": {"text": "Yes", "value": yes_value}
    })

# 保存为 JSON 文件
output_path = "datasets/MBTI_doubled.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(final_data, f, indent=4, ensure_ascii=False)

print(f"\n📁 ✅ 全部完成，结果已保存至: {output_path}")
