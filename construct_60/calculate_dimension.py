import pandas as pd
import os

# **手动输入 `x` 变量，决定要计算的结果存储路径**
x = input("请输入要计算的测试名称（对应 results/x 文件夹）: ").strip()

# 结果存储目录
results_dir = f"results/{x}"
scores_file = os.path.join(results_dir, "mbti_scores.csv")

if not os.path.exists(scores_file):
    print(f"❌ 错误：未找到 {scores_file}，请先运行测试代码获取原始评分！")
    exit(1)

# 读取数据
df = pd.read_csv(scores_file)

# **计算 Adjusted Score**
df["Adjusted Score"] = df.apply(
    lambda row: row["Score"] if row["polarity"] == 1 else (6 - row["Score"]), axis=1
)

# **计算 MBTI 维度得分**
mbti_scores = {0: 0, 1: 0, 2: 0, 3: 0}
mbti_counts = {0: 0, 1: 0, 2: 0, 3: 0}

for _, row in df.iterrows():
    dim = row["dimension"]
    mbti_scores[dim] += row["Adjusted Score"]
    mbti_counts[dim] += 1

# **计算平均得分**
for dim in mbti_scores:
    if mbti_counts[dim] > 0:
        mbti_scores[dim] /= mbti_counts[dim]

# **确定最终 MBTI 结果**
mbti_result = ""
mbti_result += "E" if mbti_scores[0] >= 3 else "I"
mbti_result += "N" if mbti_scores[1] >= 3 else "S"
mbti_result += "T" if mbti_scores[2] >= 3 else "F"
mbti_result += "J" if mbti_scores[3] >= 3 else "P"

# **存储计算后的 MBTI 结果**
mbti_result_file = os.path.join(results_dir, "mbti_result.txt")
with open(mbti_result_file, "w") as f:
    f.write(f"Model MBTI Type: {mbti_result}\n")
    f.write(f"E-I Score: {mbti_scores[0]:.2f}\n")
    f.write(f"N-S Score: {mbti_scores[1]:.2f}\n")
    f.write(f"T-F Score: {mbti_scores[2]:.2f}\n")
    f.write(f"J-P Score: {mbti_scores[3]:.2f}\n")

print(f"✅ 计算完成，MBTI 结果已保存至: {mbti_result_file}")
print(f"📄 **MBTI 结果**{mbti_result}")
print(f"📊 **MBTI 维度得分**")
print(f"E-I Score: {mbti_scores[0]:.2f}")
print(f"N-S Score: {mbti_scores[1]:.2f}")
print(f"T-F Score: {mbti_scores[2]:.2f}")
print(f"J-P Score: {mbti_scores[3]:.2f}")
