import torchaudio

"""
该文件用来测试音频是否能够被正确分割为 30 秒的片段。
"""

# 读取音频文件
audio_input, sample_rate = torchaudio.load("m.wav")

# 检查音频是否为双声道
print(f"原始音频形状：{audio_input.shape}")  # 应该显示 [2, n]，其中 2 表示双声道

# 将音频转换为单声道（取平均）
if audio_input.shape[0] == 2:  # 检查是否为双声道
    audio_input = audio_input.mean(dim=0)

# 将音频数据转换为 numpy 数组并获取长度
audio_input = audio_input.squeeze()
print(f"单声道音频长度（采样点数）：{audio_input.shape[0]}")

# 分割音频
def split_audio(audio, chunk_length_s=30, sample_rate=16000):
    chunk_length = int(chunk_length_s * sample_rate)
    total_length = audio.shape[0]
    chunks = []
    for i in range(0, total_length, chunk_length):
        chunk = audio[i:i + chunk_length]
        chunks.append(chunk)
    return chunks

audio_chunks = split_audio(audio_input, chunk_length_s=30, sample_rate=16000)

# 验证分割效果
print(f"预计分割的片段数：{len(audio_chunks)}")
for idx, chunk in enumerate(audio_chunks):
    print(f"片段 {idx + 1} 长度（采样点数）：{chunk.shape[0]}")
