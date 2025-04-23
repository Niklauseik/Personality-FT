import json
import os
import random
import tiktoken

RAW_DATA_DIR = "datasets/mbti_raw"
OUTPUT_DIR = "datasets/mbti_dpo"
os.makedirs(OUTPUT_DIR, exist_ok=True)

tokenizer = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

def build_decision_dpo(decision_type: str, sample_size: int = 800):
    decision_type = decision_type.upper()
    assert decision_type in {"T", "F"}, "输入只能是 'T' 或 'F'"

    preferred_file = "en_decision_thinking.json" if decision_type == "T" else "en_decision_feeling.json"
    non_preferred_file = "en_decision_feeling.json" if decision_type == "T" else "en_decision_thinking.json"
    label = "thinking" if decision_type == "T" else "feeling"

    with open(os.path.join(RAW_DATA_DIR, preferred_file), "r", encoding="utf-8") as f:
        preferred_data = json.load(f)
    with open(os.path.join(RAW_DATA_DIR, non_preferred_file), "r", encoding="utf-8") as f:
        non_preferred_data = json.load(f)

    assert len(preferred_data) == len(non_preferred_data), "preferred 与 non_preferred 样本数不一致"

    paired_data = list(zip(preferred_data, non_preferred_data))
    sampled = random.sample(paired_data, min(sample_size, len(paired_data)))

    all_pairs = []
    total_tokens = 0

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

    output_file = os.path.join(OUTPUT_DIR, f"{label}_{len(all_pairs)}.jsonl")
    with open(output_file, "w", encoding="utf-8") as f:
        for item in all_pairs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ 构建完成: {label}_{len(all_pairs)}.jsonl ({total_tokens} tokens)")

# ✅ 示例调用
if __name__ == "__main__":
    build_decision_dpo("F", sample_size=1600)
