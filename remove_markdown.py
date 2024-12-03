import re

def remove_markdown(text):
    """
    去除给定文本中的 Markdown 格式。

    Args:
        text (str): 含有 Markdown 格式的文本。

    Returns:
        str: 去除了 Markdown 格式的纯文本。
    """
    # 去除链接
    text = re.sub(r'$begin:math:display$([^$end:math:display$]+)\]$begin:math:text$([^)]+)$end:math:text$', r'\1', text)
    # 去除图片
    text = re.sub(r'!$begin:math:display$([^$end:math:display$]*)\]$begin:math:text$([^)]+)$end:math:text$', r'', text)
    # 去除粗体和斜体
    text = re.sub(r'(\*\*|__)(.*?)\1', r'\2', text)
    text = re.sub(r'(\*|_)(.*?)\1', r'\2', text)
    # 去除代码块和行内代码
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)  # 多行代码块
    text = re.sub(r'`([^`]*)`', r'\1', text)  # 单行代码
    # 去除标题
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    # 去除横线
    text = re.sub(r'(\*|-|_){3,}', '', text)
    # 去除表格分隔符
    text = re.sub(r'\|', '', text)
    # 去除 HTML 标签
    text = re.sub(r'<[^>]+>', '', text)
    # 去除列表项
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
    # 去除其他多余的 Markdown 转义字符
    text = re.sub(r'\\([#*`~\-+_{}[\]()>])', r'\1', text)
    # 去除多余的空行和空格
    text = re.sub(r'\n+', '\n', text)

    return text.strip()


if __name__ == "__main__":
    markdown_text = """
    # 标题

    这是一个含有 **Markdown** 的文本：
    - 列表项 1
    - 列表项 2

    还有一个链接：[点击这里](https://example.com)。

    ```python
    # 这是代码块
    print("Hello, World!")
    ```

    还有一张图片：
    ![图片描述](https://example.com/image.png)
    """

    print("原始文本:")
    print(markdown_text)
    print("\n去除 Markdown 格式后的文本:")
    plain_text = remove_markdown(markdown_text)
    print(plain_text)