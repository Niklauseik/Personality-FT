import sys
import os

# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# MBTI_test.py 所在目录
sys.path.append(current_dir)  # 确保 Python 可以找到 MBTI_test.py

# 重新导入
from MBTI_test import run_mbti_test  # ✅ 直接调用函数

import os
import pandas as pd
import collections
import time

# **手动输入 `x` 变量，决定结果存储路径**
x = input("请输入测试名称（结果将存入 results/x 文件夹）： ").strip()

# **测试参数**
NUM_TRIALS = 25  # 进行 25 次测试
RESULTS_DIR = f"results/{x}"  # 结果存储目录
os.makedirs(RESULTS_DIR, exist_ok=True)

# **存储所有测试结果**
all_predictions = []
mbti_count = collections.Counter()
dimension_counts = {dim: collections.Counter() for dim in ["E/I", "S/N", "T/F", "J/P"]}

print(f"🚀 开始 MBTI 多次测试 (共 {NUM_TRIALS} 次)...\n")

for i in range(NUM_TRIALS):
    print(f"🔄 运行测试 {i+1}/{NUM_TRIALS}...")

    # **运行测试**
    mbti_result, predictions = run_mbti_test()
    
    if mbti_result:
        mbti_count[mbti_result] += 1
        dimension_counts["E/I"][mbti_result[0]] += 1
        dimension_counts["S/N"][mbti_result[1]] += 1
        dimension_counts["T/F"][mbti_result[2]] += 1
        dimension_counts["J/P"][mbti_result[3]] += 1
        all_predictions.extend(predictions)
    else:
        print(f"⚠️ 测试 {i+1} 失败，跳过")

    time.sleep(1)

# **计算最终 MBTI 类型**
most_common_mbti = mbti_count.most_common(1)[0][0] if mbti_count else "N/A"

most_common_by_dimension = "".join([
    max(dimension_counts["E/I"], key=dimension_counts["E/I"].get, default="?"),
    max(dimension_counts["S/N"], key=dimension_counts["S/N"].get, default="?"),
    max(dimension_counts["T/F"], key=dimension_counts["T/F"].get, default="?"),
    max(dimension_counts["J/P"], key=dimension_counts["J/P"].get, default="?"),
])

# **汇总结果**
final_results = f"""
✅ MBTI 多次测试完成 (共 {NUM_TRIALS} 次)

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

# **保存最终结果**

final_result_file = os.path.join(RESULTS_DIR, "final_mbti_results.txt")
with open(final_result_file, "w") as f:
    f.write(final_results)

print(final_results)
print(f"📁 结果已保存至: {final_result_file}")
