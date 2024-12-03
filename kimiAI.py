from openai import OpenAI


def summarize_class_content(user_provided_text, max_length=8192):
    """
    使用 Kimi AI API 对课堂内容进行总结。

    Args:
        api_key (str): 用于调用 API 的密钥。
        user_provided_text (str): 课堂内容文本。
        max_length (int, optional): 文本长度限制，防止超过 API 的令牌限制。默认值为 8192。

    Returns:
        str: 生成的课堂总结。
    """
    client = OpenAI(
        api_key="**********",  # API 密钥
        base_url="https://api.moonshot.cn/v1",
    )

    # 精简系统描述
    system_message = {
        "role": "system",
        "content": "你是一个课堂总结助手，为用户生成清晰简洁的课堂内容总结。"
    }

    # 截取长文本
    trimmed_text = user_provided_text[:max_length]

    # 调用 API 生成总结
    completion = client.chat.completions.create(
        model="moonshot-v1-8k",
        messages=[
            system_message,
            {"role": "user", "content": f"以下是课堂内容：{trimmed_text}"},
        ],
        temperature=0.3,
    )

    # 提取并返回总结
    return completion.choices[0].message.content


# 示例调用
if __name__ == "__main__":
    user_text = "今天我们学习了二次函数的基本性质，了解了顶点、对称轴以及开口方向的概念。" * 200
    summary = summarize_class_content(user_text)
    print("课堂总结：", summary)
