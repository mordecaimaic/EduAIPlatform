# macos代码
# 下面的代码需要pytorch因为使用到了mps
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import os  # 添加此行以设置环境变量
import time
from prepare_audio import prepare_audio
import glob

# 这行代码的作用：设置环境变量 PYTORCH_ENABLE_MPS_FALLBACK 为 1
# 这将会启用 CPU 回退机制：设置 PYTORCH_ENABLE_MPS_FALLBACK=1 后，

# PyTorch 会在 MPS 设备上遇到未实现的操作时，自动将该操作转移到 CPU 上执行，而不是抛出错误。
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
# 将 Hugging Face 模型的缓存目录设置为相对路径
custom_cache_dir = "./models"

# 模型参数设置
'''
    模型名称 可以切换为 "openai/whisper-large-v3" 或者 openai/whisper-small" ）
    目前只安装了small模型和large模型
    由于设置了custom_cache_dir，所以模型会被下载到custom_cache_dir的目录下，也就是当前目录的相对路径
'''
model_name = "openai/whisper-large-v3"  # 模型名称
# model_name = "openai/whisper-small"  # 模型名称
language = "chinese"  # 设置为语言代码，例如 "zh"，或"cantonese" None 以自动检测
task = "transcribe"  # "transcribe" 或 "translate"

# 获取设备的类别
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# 加载模型和处理器
processor = WhisperProcessor.from_pretrained(
    model_name,
    language=language,
    task=task,
    local_files_only=True,
    cache_dir=custom_cache_dir  # 设置缓存目录
)
model = WhisperForConditionalGeneration.from_pretrained(
    model_name,
    local_files_only=True,
    cache_dir=custom_cache_dir  # 设置缓存目录
).to(device)

# 添加新的 <pad> 标记，并确保与 eos_token 不同
if processor.tokenizer.pad_token is None or processor.tokenizer.pad_token == processor.tokenizer.eos_token:
    processor.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(processor.tokenizer), mean_resizing=False)  # 设置 mean_resizing=False
    processor.tokenizer.pad_token_id = processor.tokenizer.convert_tokens_to_ids('[PAD]')
    model.config.pad_token_id = processor.tokenizer.pad_token_id

# 输出 pad_token_id 和 eos_token_id 以验证它们是否不同
print("pad_token_id:", processor.tokenizer.pad_token_id)
print("eos_token_id:", processor.tokenizer.eos_token_id)

# 读取音频文件
# 使用封装好的 prepare_audio 函数加载和预处理音频
audio_input, sample_rate = prepare_audio("audios/m.mp3")

# 将音频数据转换为 numpy 数组并扁平化
audio_input = audio_input.squeeze().numpy()


# 定义函数，将音频拆分为 30 秒的片段
def split_audio(audio, chunk_length_s=30, sample_rate=16000):
    chunk_length = int(chunk_length_s * sample_rate)
    total_length = audio.shape[0]
    chunks = []
    for i in range(0, total_length, chunk_length):
        chunk = audio[i:i + chunk_length]
        chunks.append(chunk)
    return chunks


# 拆分音频
audio_chunks = split_audio(audio_input, chunk_length_s=30, sample_rate=16000)
print(f"音频被拆分为 {len(audio_chunks)} 个片段")

# 处理每个音频片段
transcriptions = []
for idx, chunk in enumerate(audio_chunks):
    # 处理音频输入
    input_features = processor(
        chunk,
        sampling_rate=16000,
        return_tensors="pt"
    ).input_features.to(device)

    # 创建 attention_mask
    attention_mask = torch.ones(
        input_features.shape[:-1],
        dtype=torch.long
    ).to(device)

    # 获取强制解码器 ID
    if language is not None:
        forced_decoder_ids = processor.get_decoder_prompt_ids(
            language=language,
            task=task
        )
    else:
        forced_decoder_ids = processor.get_decoder_prompt_ids(
            task=task
        )

    # 生成文本
    predicted_ids = model.generate(
        input_features,
        attention_mask=attention_mask,  # 添加 attention_mask
        forced_decoder_ids=forced_decoder_ids
    )
    transcription = processor.decode(
        predicted_ids[0],
        skip_special_tokens=True
    )

    # 保存转录结果
    transcriptions.append(transcription)
    print(f"片段 {idx + 1}/{len(audio_chunks)} 转录完成: " + transcription)

# 合并所有转录结果
full_transcription = ' '.join(transcriptions)

print("完整转录结果：")
for transcription in transcriptions:
    print(transcription)
