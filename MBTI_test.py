import openai
import pandas as pd
import time
import collections

# 读取 OpenAI API Key
from utils import config_manager
config_manager = config_manager.ConfigManager()
api_key = config_manager.get_api_key("openai")

# 初始化 OpenAI 客户端
client = openai.OpenAI(api_key=api_key)

# 读取 MBTI 问题数据集
MBTI_DATASET = "datasets/MBTI_doubled.json"  # ✅ 使用 double 数据

def run_mbti_test():
    """
    运行 MBTI 测试，返回模型预测的 MBTI 结果和所有 a/b 选择。
    """

    # **读取 JSON 数据**
    df = pd.read_json(MBTI_DATASET)

    # **确保数据格式正确**
    if "question" not in df.columns or "choice_a" not in df.columns or "choice_b" not in df.columns:
        raise ValueError("数据集格式错误，缺少 `question`, `choice_a`, `choice_b` 字段")

    # **构造 System Prompt**
    system_message = (
        "You are answering an MBTI personality test.\n"
        "For the question, select either choice_a or choice_b.\n"
        "Respond with only one word: 'a' or 'b'."
    )

    predictions = []
    dimension_count = collections.Counter({"E": 0, "I": 0, "S": 0, "N": 0, "T": 0, "F": 0, "J": 0, "P": 0})  # 记录 MBTI 维度

    for idx, row in df.iterrows():
        question = row["question"]
        choice_a_text = row["choice_a"]["text"] if isinstance(row["choice_a"], dict) else ""
        choice_b_text = row["choice_b"]["text"] if isinstance(row["choice_b"], dict) else ""

        choice_a_value = row["choice_a"]["value"] if isinstance(row["choice_a"], dict) else ""
        choice_b_value = row["choice_b"]["value"] if isinstance(row["choice_b"], dict) else ""

        query = f"Q{idx+1}: {question}\nA: {choice_a_text} (a)\nB: {choice_b_text} (b)"

        retry_count = 0
        success = False

        while retry_count < 3 and not success:
            try:
                response = client.chat.completions.create(
                    model="ft:gpt-4o-mini-2024-07-18:personal:isfp-general:BBBpCFwm",
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": query}
                    ]
                )

                model_response = response.choices[0].message.content.strip().lower()
                print(f"🤖 {model_response}")

                # **检查返回格式**
                if model_response not in {"a", "b"}:
                    raise ValueError(f"❌ 发现无效答案: {model_response}")

                # **存入预测结果**
                predictions.append(model_response)

                # **更新 MBTI 维度计数**
                if model_response == "a":
                    dimension_count[choice_a_value] += 1
                else:
                    dimension_count[choice_b_value] += 1

                success = True

            except Exception as e:
                print(f"⚠️ API 调用失败（尝试 {retry_count + 1}/3），错误: {e}")
                retry_count += 1
                time.sleep(1)  # 等待 1 秒再重试

    # **计算最终 MBTI 类型**
    mbti_result = ""
    mbti_result += "E" if dimension_count["E"] >= dimension_count["I"] else "I"
    mbti_result += "S" if dimension_count["S"] >= dimension_count["N"] else "N"
    mbti_result += "T" if dimension_count["T"] >= dimension_count["F"] else "F"
    mbti_result += "J" if dimension_count["J"] >= dimension_count["P"] else "P"

    return mbti_result, predictions  # ✅ 返回最终 MBTI 类型 + 详细选项记录

if __name__ == "__main__":
    mbti_result, choices = run_mbti_test()

    if mbti_result:
        print(f"📊 **模型预测的 MBTI 类型: {mbti_result}**")
        print(f"📌 详细回答 (a/b): {choices}")
    else:
        print("❌ 预测失败！")
