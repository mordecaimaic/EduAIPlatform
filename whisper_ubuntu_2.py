import textwrap

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from scipy.signal import resample
import soundfile as sf
import os

# 设置设备
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# 模型路径
model_path = "/data/disk3/mqc/whisper/models/openai/whisper-large-v3"

# 加载模型和处理器
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_path,
    torch_dtype=torch_dtype,
    local_files_only=True
).to(device)

processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)

# 添加特殊 token 设置，确保 pad_token 和 eos_token 不同
if processor.tokenizer.pad_token is None or processor.tokenizer.pad_token == processor.tokenizer.eos_token:
    processor.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(processor.tokenizer))
    processor.tokenizer.pad_token_id = processor.tokenizer.convert_tokens_to_ids('[PAD]')
    model.config.pad_token_id = processor.tokenizer.pad_token_id

# 设置环境变量限制线程数
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# 加载音频
audio_path = "audios/m.wav"
audio_input, sample_rate = sf.read(audio_path)

# 转换为单通道
if len(audio_input.shape) > 1:
    audio_input = audio_input.mean(axis=1)

# 重新采样为 16kHz
target_sample_rate = 16000
if sample_rate != target_sample_rate:
    audio_input = resample(audio_input, int(len(audio_input) * target_sample_rate / sample_rate))

# 定义音频拆分函数
def split_audio(audio, chunk_length_s=30, sample_rate=16000, overlap=5):
    chunk_length = int(chunk_length_s * sample_rate)
    overlap_length = int(overlap * sample_rate)
    total_length = len(audio)
    chunks = []
    for i in range(0, total_length, chunk_length - overlap_length):
        chunk = audio[i:i + chunk_length]
        chunks.append(chunk)
    return chunks

# 拆分音频
audio_chunks = split_audio(audio_input, chunk_length_s=30, overlap=5)
print(f"音频被拆分为 {len(audio_chunks)} 个片段")

# 转录逻辑
transcriptions = []
for idx, chunk in enumerate(audio_chunks):
    # 转换为输入特征
    inputs = processor(
        chunk,
        sampling_rate=target_sample_rate,
        return_tensors="pt",
        padding=True  # 确保批处理时自动填充
    )
    input_features = inputs.input_features.to(device)

    # 手动生成 attention_mask，如果不可用
    attention_mask = torch.ones(input_features.shape[:-1], dtype=torch.long, device=device)

    # 设置语言和任务（强制中文解码）
    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="zh",  # 显式设置为中文
        task="transcribe"
    )

    # 模型生成
    predicted_ids = model.generate(
        input_features,
        attention_mask=attention_mask,  # 显式传递 attention_mask
        max_length=1000,
        num_beams=5,
        forced_decoder_ids=forced_decoder_ids
    )

    # 解码结果
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    transcriptions.append(transcription[0])
    print(f"片段 {idx + 1}/{len(audio_chunks)} 转录完成：{transcription[0]}")

# 合并并去重拼接
def clean_and_merge_transcriptions(transcriptions, overlap_s=5):
    merged_transcription = ""
    overlap_text = ""
    for idx, text in enumerate(transcriptions):
        if idx > 0 and overlap_text in text:
            text = text.replace(overlap_text, "").strip()
        overlap_text = " ".join(text.split()[-overlap_s:])  # 获取最后几句作为重叠参考
        merged_transcription += text + " "
    return merged_transcription.strip()

full_transcription = clean_and_merge_transcriptions(transcriptions, overlap_s=5)

print("完整转录结果：", full_transcription)
# 设置每行显示的最大宽度
max_width = 80

# 使用 textwrap 将长字符串拆分为多行
wrapped_transcription = textwrap.fill(full_transcription, width=max_width)

print("完整转录结果：\n", wrapped_transcription)