import json

# 加载两个 JSON 文件
with open("datasets/MBTI_doubled_4omini.json", "r", encoding="utf-8") as f1, \
     open("datasets/MBTI_doubled_nano.json", "r", encoding="utf-8") as f2:
    data_gpt = json.load(f1)
    data_nano = json.load(f2)

# 比较差异
differences = []
for item1, item2 in zip(data_gpt, data_nano):
    if (
        item1["question"] != item2["question"] or
        item1["choice_a"]["text"] != item2["choice_a"]["text"] or
        item1["choice_a"]["value"] != item2["choice_a"]["value"] or
        item1["choice_b"]["value"] != item2["choice_b"]["value"]
    ):
        differences.append({
            "question": item1["question"],
            "gpt_choice_a": item1["choice_a"],
            "gpt_choice_b": item1["choice_b"],
            "nano_choice_a": item2["choice_a"],
            "nano_choice_b": item2["choice_b"]
        })

# 保存差异结果
output_path = "datasets/differences_gpt_vs_nano.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(differences, f, indent=4, ensure_ascii=False)

print(f"✅ 差异比较完成，发现 {len(differences)} 项不同，结果保存至 {output_path}")
