import torchaudio


def prepare_audio(file_path, target_sample_rate=16000):
    """
    加载并预处理音频文件，确保音频为单声道且采样率为目标采样率。

    参数:
    - file_path (str): 音频文件路径
    - target_sample_rate (int): 目标采样率，默认为 16000 Hz

    返回:
    - audios (Tensor): 预处理后的音频数据，单声道，目标采样率
    - sample_rate (int): 采样率，等于目标采样率
    """
    # 读取音频文件
    audio, sample_rate = torchaudio.load(file_path)

    # 如果音频是双声道，将其转换为单声道
    if audio.shape[0] == 2:
        audio = audio.mean(dim=0)

    # 如果采样率不是目标采样率(16000HZ)，则进行重采样
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        audio = resampler(audio)

    return audio.squeeze(), target_sample_rate
