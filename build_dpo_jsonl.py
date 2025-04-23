import json
import os
import random
import tiktoken

# ✅ MBTI四维及其对应文件名
MBTI_DIMENSIONS = {
    0: ("E", "I", "energy_extraversion", "energy_introversion"),
    1: ("N", "S", "information_intuition", "information_sensing"),
    2: ("T", "F", "decision_thinking", "decision_feeling"),
    3: ("J", "P", "execution_judging", "execution_perceiving")
}

RAW_DATA_DIR = "datasets/mbti_raw"
OUTPUT_DIR = "datasets/mbti_dpo"
os.makedirs(OUTPUT_DIR, exist_ok=True)

tokenizer = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

def build_dpo_dataset(mbti_type: str, sample_size: int = 800):
    mbti_type = mbti_type.upper()
    assert len(mbti_type) == 4, "MBTI type must be 4 letters (e.g., ENTJ)"

    all_pairs = []
    total_tokens = 0

    for i, (pos_letter, neg_letter, pos_file, neg_file) in MBTI_DIMENSIONS.items():
        letter = mbti_type[i]
        if letter == pos_letter:
            preferred_file = f"en_{pos_file}.json"
            non_preferred_file = f"en_{neg_file}.json"
        else:
            preferred_file = f"en_{neg_file}.json"
            non_preferred_file = f"en_{pos_file}.json"

        # 读取数据
        with open(os.path.join(RAW_DATA_DIR, preferred_file), "r", encoding="utf-8") as f:
            preferred_data = json.load(f)
        with open(os.path.join(RAW_DATA_DIR, non_preferred_file), "r", encoding="utf-8") as f:
            non_preferred_data = json.load(f)

        assert len(preferred_data) == len(non_preferred_data), f"{preferred_file} 和 {non_preferred_file} 行数不一致"

        # 对该维度分别采样 sample_size 条
        paired_data = list(zip(preferred_data, non_preferred_data))
        sampled = random.sample(paired_data, min(sample_size, len(paired_data)))

        for p, np in sampled:
            item = {
                "input": {
                    "messages": [
                        {"role": "user", "content": p["instruction"]}
                    ]
                },
                "preferred_output": [
                    {"role": "assistant", "content": p["output"]}
                ],
                "non_preferred_output": [
                    {"role": "assistant", "content": np["output"]}
                ]
            }
            all_pairs.append(item)
            total_tokens += (
                count_tokens(p["instruction"]) +
                count_tokens(p["output"]) +
                count_tokens(np["output"])
            )

    # 写入 jsonl
    output_file = os.path.join(OUTPUT_DIR, f"{mbti_type}_{len(all_pairs)}.jsonl")
    with open(output_file, "w", encoding="utf-8") as f:
        for item in all_pairs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ {mbti_type}: 每维度各采样 {sample_size}，共计 {len(all_pairs)} 条，约 {total_tokens} tokens ➜ {output_file}")

# ✅ 示例调用
if __name__ == "__main__":
    build_dpo_dataset("ENTJ", sample_size=1600)
