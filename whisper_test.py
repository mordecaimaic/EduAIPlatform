import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio

# 加载模型和处理器，并指定语言为英语
# 使用large模型时，需要更大的内存
# processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3", language="en", task="transcribe")
# model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3").to("mps")

processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="en", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to("mps")

# 读取音频文件
audio_input, sample_rate = torchaudio.load("audio1.mp3")

# 如果采样率不是16000Hz，需要重新采样
if sample_rate != 16000:
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
    audio_input = resampler(audio_input)

# 将音频数据转换为numpy数组并扁平化
audio_input = audio_input.squeeze().numpy()

# 处理音频输入
input_features = processor(audio_input, sampling_rate=16000, return_tensors="pt").input_features.to("mps")

# 获取强制解码器ID以指定语言
forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")

# 生成文本，指定forced_decoder_ids
predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
transcription = processor.decode(predicted_ids[0], skip_special_tokens=True)

print(transcription)
