import os
import json

# 设定数据目录
SAMPLES_DIR = "datasets/mbti_samples"
OUTPUT_DIR = "datasets/mbti_ft"

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 用户输入目标 MBTI 性格
mbti_type = input("请输入 MBTI 性格类型（如 ENTJ, ISFP 等）：").strip().upper()

# MBTI 维度对应的数据集
MBTI_TO_DATASET = {
    "E": "en_energy_extraversion.json",
    "I": "en_energy_introversion.json",
    "S": "en_information_sensing.json",
    "N": "en_information_intuition.json",
    "T": "en_decision_thinking.json",
    "F": "en_decision_feeling.json",
    "J": "en_execution_judging.json",
    "P": "en_execution_perceiving.json",
}

# 验证输入
if len(mbti_type) != 4 or any(c not in MBTI_TO_DATASET for c in mbti_type):
    print("❌ 输入错误，请输入有效的 MBTI 类型（如 ENTJ, INFP）")
    exit(1)

# 选择对应数据集
selected_files = [MBTI_TO_DATASET[c] for c in mbti_type]
print(f"📌 选择的数据集: {selected_files}")

# **合并数据**
combined_data = []
for file_name in selected_files:
    file_path = os.path.join(SAMPLES_DIR, file_name)
    
    if not os.path.exists(file_path):
        print(f"⚠️ 文件缺失: {file_name}，跳过该类别")
        continue
    
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

        # **转换为 `messages` 格式**
        for item in data:
            # 移除 input，如果为空
            user_content = item["instruction"]
            if item.get("input") and item["input"].strip():
                user_content += "\n" + item["input"]

            # 组装 OpenAI 微调格式
            messages = [
                {"role": "system", "content": "You are an AI assistant with a strong, distinctive personality."},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": item["output"]}
            ]
            combined_data.append({"messages": messages})

# **保存为 JSONL**
output_file = os.path.join(OUTPUT_DIR, f"{mbti_type}_general.jsonl")
with open(output_file, "w", encoding="utf-8") as f:
    for entry in combined_data:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"✅ 训练数据集已保存至: {output_file}，共 {len(combined_data)} 条数据（JSONL 格式）")
