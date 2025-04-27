import json
import os

# **读取 MBTI 原始数据**
input_file = "datasets/MBTI.json"
output_file = "datasets/MBTI_doubled.json"

# 确保数据目录存在
os.makedirs(os.path.dirname(output_file), exist_ok=True)

with open(input_file, "r", encoding="utf-8") as f:
    original_mbti_data = json.load(f)

# **创建互换版本**
doubled_mbti_data = []
for item in original_mbti_data:
    # 原始版本
    original = {
        "question": item["question"],
        "choice_a": item["choice_a"],
        "choice_b": item["choice_b"]
    }
    # 互换 A/B 版本
    reversed_item = {
        "question": item["question"],
        "choice_a": item["choice_b"],  # 互换 A/B
        "choice_b": item["choice_a"]
    }
    doubled_mbti_data.extend([original, reversed_item])

# **保存翻倍后的数据**
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(doubled_mbti_data, f, indent=4, ensure_ascii=False)

print(f"✅ 处理完成，数据翻倍，已保存至: {output_file}")
print(f"📊 原始数据: {len(original_mbti_data)} 条")
print(f"📊 处理后数据: {len(doubled_mbti_data)} 条 (翻倍)")
