import json
import csv

# 读取 questions.txt（单行 JSON-like 数据）
with open("datasets/questions.txt", "r", encoding="utf-8") as f:
    raw_data = f.read().strip()

# 处理 HTML 转义字符（`&quot;` -> `"`）
raw_data = raw_data.replace("&quot;", "\"")

# 转换为 Python 可解析的 JSON 格式
parsed_data = json.loads(raw_data)

# 提取所有问题
questions = [item["text"] for item in parsed_data]

# 写入 CSV 文件
csv_filename = "mbti_questions.csv"
with open(csv_filename, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Question"])  # 添加表头
    for question in questions:
        writer.writerow([question])

print(f"✅ 成功转换 {len(questions)} 道 MBTI 题，并保存到 `{csv_filename}`")

# 读取并清理 CSV 文件
csv_filename = "mbti_questions.csv"

with open(csv_filename, "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    data = [row[0].strip().replace('"', '') for row in reader if row]  # 去除双引号

# 直接覆盖原 CSV 文件
with open(csv_filename, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Question"])  # 重新写入表头
    for question in data[1:]:  # 跳过原始表头
        writer.writerow([question])

print(f"✅ 已去除双引号，并直接覆盖 `{csv_filename}`")
