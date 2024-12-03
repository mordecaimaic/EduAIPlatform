import os
import torch
from transformers import MBartForConditionalGeneration, MBart50Tokenizer

# 设置自定义模型缓存路径
custom_cache_dir = "./models"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # 启用 MPS Fallback，用于 macOS M1/M2 设备

# 模型参数
model_name = "facebook/mbart-large-50-many-to-many-mmt"  # mBART 模型名称

# 检查设备类型
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# 加载分词器和模型，并设置缓存目录
tokenizer = MBart50Tokenizer.from_pretrained(
    model_name,
    cache_dir=custom_cache_dir,  # 设置缓存目录
    src_lang="zh_CN"            # 指定源语言为中文
)
model = MBartForConditionalGeneration.from_pretrained(
    model_name,
    cache_dir=custom_cache_dir,  # 设置缓存目录
    local_files_only=True       # 如果模型未下载，则从 Hugging Face 自动下载
).to(device)

# 示例文本：课堂转录
input_text = """
今天的课堂讲解了深度学习的基础知识，包括神经网络的多层感知机结构、激活函数的作用及其选择。
随后，我们讨论了反向传播算法及其在训练过程中的重要性。
最后，我们实现了一个简单的二分类问题的神经网络，并进行了实验分析。
"""

# 编码输入文本
inputs = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True).to(device)

# 添加目标语言
tokenizer.tgt_lang = "zh_CN"  # 指定目标语言为简体中文

# 生成摘要
summary_ids = model.generate(
    inputs["input_ids"],
    max_length=150,
    min_length=50,
    length_penalty=2.0,
    num_beams=4
)

# 解码生成的摘要
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print("生成的课堂总结：")
print(summary)