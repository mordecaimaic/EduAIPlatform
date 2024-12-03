import os
import torch
from transformers import BartForConditionalGeneration, BertTokenizer

# 设置自定义模型缓存路径
custom_cache_dir = "./models"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# 模型参数
model_name = custom_cache_dir + "/" + "fnlp/bart-base-chinese"

# 检查设备类型
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# 加载分词器和模型
tokenizer = BertTokenizer.from_pretrained(
    model_name,
    cache_dir=custom_cache_dir,
    local_files_only=True  # 如果模型未下载，则屏蔽这行自动从 Hugging Face 下载
)
model = BartForConditionalGeneration.from_pretrained(
    model_name,
    cache_dir=custom_cache_dir,
    local_files_only=True
).to(device)

# 示例文本：课堂转录
input_text = """
自适应知识蒸馏：在完成领域和任务自适应微调后，可以通过自适应知识蒸馏将该特定知识传递给DistilBERT。此过程利用微调后的BERT作为教师模型，通过知识蒸馏将领域和任务的专业知识浓缩到DistilBERT中。这样可以在保留DistilBERT的轻量化优势的同时还提升了其在特定课堂任务和领域的表现。

动态权重调整机制：师生上课课堂内容以及形式都是包含不同场景且形式多样化，可能涉及讲解、提问、讨论等多种情境。引入动态权重调整机制，使模型能够根据内容的类型动态调整各模块的权重，以更精准地捕捉课堂的重点内容。例如当检测到提问或讨论内容时，权重可以动态调整以提升对互动内容的关注度；而在讲解中则聚焦于知识点的提取[26]。这种动态权重机制可以帮助模型在复杂课堂环境中更灵活地平衡细节和重点内容的提取，提高内容生成的适用性和准确性。

多头注意力机制：引入多头注意力机制的优化可以增强模型在长文本和多主题切换环境中的适应性。多头注意力机制通过不同注意力头捕捉不同文本层次上的细节信息，使得模型对复杂内容和主题切换的理解力得到显著提升。"""

# 编码输入文本
inputs = tokenizer.encode_plus(
    input_text.strip(),
    return_tensors="pt",
    max_length=512,
    truncation=True
).to(device)

# 生成摘要
summary_ids = model.generate(
    inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_length=150,
    min_length=50,
    length_penalty=2.0,
    num_beams=4,
    no_repeat_ngram_size=3
)

# 解码生成的摘要
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print("生成的课堂总结：")
print(summary)
