import os
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from prepare_audio import prepare_audio


def transcribe_audio(audio_path, model_name="openai/whisper-large-v3", language="chinese", task="transcribe"):
    """
    转录音频为文本。

    参数:
    - audio_path (str): 音频文件路径。
    - model_name (str): Whisper 模型名称。
    - language (str): 转录的目标语言。
    - task (str): 转录任务类型 ("transcribe" 或 "translate")。

    返回:
    - str: 转录的文本结果。
    """
    # 设置环境变量以支持 MPS 回退
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    # 定义模型缓存目录
    custom_cache_dir = "./models"

    # 选择设备
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # 加载处理器和模型
    processor = WhisperProcessor.from_pretrained(
        model_name,
        language=language,
        task=task,
        local_files_only=True,
        cache_dir=custom_cache_dir
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        model_name,
        local_files_only=True,
        cache_dir=custom_cache_dir
    ).to(device)

    # 确保处理器包含 pad_token
    if processor.tokenizer.pad_token is None or processor.tokenizer.pad_token == processor.tokenizer.eos_token:
        processor.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(processor.tokenizer), mean_resizing=False)
        processor.tokenizer.pad_token_id = processor.tokenizer.convert_tokens_to_ids('[PAD]')
        model.config.pad_token_id = processor.tokenizer.pad_token_id

    # 加载和预处理音频
    audio_input, sample_rate = prepare_audio(audio_path)
    audio_input = audio_input.squeeze().numpy()

    # 将音频拆分为 30 秒的片段
    def split_audio(audio, chunk_length_s=30, sample_rate=16000):
        chunk_length = int(chunk_length_s * sample_rate)
        total_length = audio.shape[0]
        chunks = [audio[i:i + chunk_length] for i in range(0, total_length, chunk_length)]
        return chunks

    audio_chunks = split_audio(audio_input, chunk_length_s=30, sample_rate=16000)

    # 转录每个片段
    transcriptions = []
    for chunk in audio_chunks:
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
        forced_decoder_ids = processor.get_decoder_prompt_ids(
            language=language,
            task=task
        )

        # 生成文本
        predicted_ids = model.generate(
            input_features,
            attention_mask=attention_mask,
            forced_decoder_ids=forced_decoder_ids
        )
        transcription = processor.decode(predicted_ids[0], skip_special_tokens=True)
        transcriptions.append(transcription)

    # 合并结果
    full_transcription = ' '.join(transcriptions)
    return full_transcription


# 示例调用
if __name__ == "__main__":
    audio_file_path = "uploads/mid_83.mp3"  # 替换为您的音频文件路径
    transcription_result = transcribe_audio(audio_file_path)
    print("转录结果：")
    print(transcription_result)



