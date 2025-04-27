import json
import os

# ✅ 配置区：直接改这里即可
input_dataset = 'thinking_1600'  # 输入数据集名（不含路径和扩展名）
output_dataset = 'feeling_1600'  # 输出数据集名（不含路径和扩展名）
base_dir = 'datasets/mbti_dpo'  # 数据集所在目录

# 自动构建文件路径
input_file = os.path.join(base_dir, input_dataset + '.jsonl')
output_file = os.path.join(base_dir, f'{output_dataset}_reversed_from_{input_dataset}.jsonl')

# 打开输入文件和输出文件
with open(input_file, 'r', encoding='utf-8') as fin, open(output_file, 'w', encoding='utf-8') as fout:
    for line in fin:
        data = json.loads(line)
        # 交换 preferred_output 和 non_preferred_output
        data['preferred_output'], data['non_preferred_output'] = data['non_preferred_output'], data['preferred_output']
        # 写入新文件
        fout.write(json.dumps(data, ensure_ascii=False) + '\n')

print(f"✅ 数据集反转完成！\n输入: {input_file}\n输出: {output_file}")