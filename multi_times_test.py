import sys
import os
import pandas as pd
import collections
import time

# ✅ 配置模型相关参数（填这里）
MODEL_TYPE = "deepseek"          # "gpt" 或 "deepseek"
MODEL_NAME = "deepseek-chat"  # 模型名称（如 gpt-4o-mini 或 deepseek-chat）
EARLY_STOP_COUNT = 15       # 出现次数达到此数值即提前停止

# ✅ 目录设置
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from MBTI_test import run_mbti_test  # ✅ 调用测试函数

# ✅ 输入测试名称
x = input("请输入测试名称（结果将存入 results/x 文件夹）： ").strip()
NUM_TRIALS = 25
RESULTS_DIR = f"results/{x}"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ✅ 初始化
all_predictions = []
mbti_count = collections.Counter()
dimension_counts = {dim: collections.Counter() for dim in ["E/I", "S/N", "T/F", "J/P"]}

print(f"🚀 开始 MBTI 多次测试（最多 {NUM_TRIALS} 次）...\n")

# ✅ 主循环
for i in range(NUM_TRIALS):
    print(f"🔄 运行测试 {i+1}/{NUM_TRIALS}...")

    mbti_result, predictions = run_mbti_test(MODEL_NAME, MODEL_TYPE)

    if mbti_result:
        mbti_count[mbti_result] += 1
        dimension_counts["E/I"][mbti_result[0]] += 1
        dimension_counts["S/N"][mbti_result[1]] += 1
        dimension_counts["T/F"][mbti_result[2]] += 1
        dimension_counts["J/P"][mbti_result[3]] += 1
        all_predictions.extend(predictions)
    else:
        print(f"⚠️ 测试 {i+1} 失败，跳过")

    # ✅ 提前停止判断
    most_common = mbti_count.most_common(1)
    if most_common and most_common[0][1] >= EARLY_STOP_COUNT:
        print(f"⏹️ 满足提前终止条件：{most_common[0][0]} 出现 {most_common[0][1]} 次，停止测试")
        break

    time.sleep(0.3)

# ✅ 结果统计
most_common_mbti = mbti_count.most_common(1)[0][0] if mbti_count else "N/A"
most_common_by_dimension = "".join([
    max(dimension_counts["E/I"], key=dimension_counts["E/I"].get, default="?"),
    max(dimension_counts["S/N"], key=dimension_counts["S/N"].get, default="?"),
    max(dimension_counts["T/F"], key=dimension_counts["T/F"].get, default="?"),
    max(dimension_counts["J/P"], key=dimension_counts["J/P"].get, default="?"),
])

# ✅ 汇总输出
final_results = f"""
✅ MBTI 多次测试完成（最多 {NUM_TRIALS} 次，实际运行 {sum(mbti_count.values())} 次）

📌 完整 MBTI 类型统计
{mbti_count}

📌 按维度统计
E/I: {dimension_counts["E/I"]}
S/N: {dimension_counts["S/N"]}
T/F: {dimension_counts["T/F"]}
J/P: {dimension_counts["J/P"]}

📊 方式 1：最多完整 MBTI 类型 → {most_common_mbti}
📊 方式 2：按维度最多 → {most_common_by_dimension}
"""

final_result_file = os.path.join(RESULTS_DIR, "final_mbti_results.txt")
with open(final_result_file, "w", encoding="utf-8") as f:
    f.write(final_results)

print(final_results)
print(f"📁 结果已保存至: {final_result_file}")
