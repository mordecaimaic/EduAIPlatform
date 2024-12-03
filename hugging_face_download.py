from huggingface_hub import snapshot_download
if __name__ == '__main__':
    # 下载整个模型到指定的缓存目录
    # snapshot_download(
    #     repo_id="openai/whisper-small",  # 下载 small 模型
    #     cache_dir="./models",
    #     endpoint="https://hf-mirror.com"
    # )
    # # 下载large模型
    snapshot_download(
        repo_id="openai/whisper-large-v3",  # 下载 large 模型
        cache_dir="./models",
        endpoint="https://hf-mirror.com"
    )