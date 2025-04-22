import os
import pandas as pd
import time
import collections
import openai

from utils import config_manager
config_manager = config_manager.ConfigManager()

# ✅ 模型映射表
MODEL_CONFIG = {
    "gpt": {
        "api_key": config_manager.get_api_key("openai"),
        "base_url": "https://api.openai.com/v1"
    },
    "deepseek": {
        "api_key": config_manager.get_api_key("deepseek"),
        "base_url": "https://api.deepseek.com"
    }
}

MBTI_DATASET = "datasets/MBTI_doubled.json"

def run_mbti_test(model_name, model_type="gpt"):
    """
    根据模型类型（gpt / deepseek）运行 MBTI 测试
    """
    df = pd.read_json(MBTI_DATASET)

    if "question" not in df.columns or "choice_a" not in df.columns or "choice_b" not in df.columns:
        raise ValueError("数据集格式错误")

    # ✅ 统一初始化 Client
    config = MODEL_CONFIG[model_type]
    client = openai.OpenAI(api_key=config["api_key"], base_url=config["base_url"])

    system_message = (
        "You are answering an MBTI personality test.\n"
        "For the question, select either choice_a or choice_b.\n"
        "Respond with only one word: 'a' or 'b'."
    )

    predictions = []
    dimension_count = collections.Counter({"E": 0, "I": 0, "S": 0, "N": 0, "T": 0, "F": 0, "J": 0, "P": 0})

    for idx, row in df.iterrows():
        question = row["question"]
        a_text = row["choice_a"]["text"]
        b_text = row["choice_b"]["text"]
        a_value = row["choice_a"]["value"]
        b_value = row["choice_b"]["value"]

        query = f"Q{idx+1}: {question}\nA: {a_text} (a)\nB: {b_text} (b)"

        for retry in range(3):
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": query}
                    ]
                )
                ans = response.choices[0].message.content.strip().lower()
                print(f"🤖 {ans}")

                if ans not in {"a", "b"}:
                    raise ValueError(f"Invalid answer: {ans}")
                predictions.append(ans)
                dimension_count[a_value if ans == "a" else b_value] += 1
                break
            except Exception as e:
                print(f"⚠️ 尝试 {retry + 1}/3 失败: {e}")
                time.sleep(1)

    mbti_result = "".join([
        "E" if dimension_count["E"] >= dimension_count["I"] else "I",
        "S" if dimension_count["S"] >= dimension_count["N"] else "N",
        "T" if dimension_count["T"] >= dimension_count["F"] else "F",
        "J" if dimension_count["J"] >= dimension_count["P"] else "P",
    ])

    return mbti_result, predictions
