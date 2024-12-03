from flask import Flask, request, jsonify, make_response
import os
from audio_transcriber import transcribe_audio
from kimiAI import summarize_class_content
from remove_markdown import remove_markdown
from time import sleep

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False  # 确保 JSON 数据以 UTF-8 格式返回

UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


@app.route('/transcribe', methods=['POST'])
def handle_audio_transcription():
    # 休眠5秒
    sleep(3)

    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # 保存文件
    save_path = os.path.join("uploads", file.filename)
    file.save(save_path)
    print(f"Received file: {save_path}")

    #  使用whisper处理音频文件
    raw_text = transcribe_audio(save_path)

    # 模拟返回内容
    result = {"message": raw_text, "filename": file.filename}

    # 设置响应头并返回
    response = make_response(jsonify(result))
    return response, 200


@app.route('/generateNote', methods=['POST'])
def handle_audio_note():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # 保存文件
    save_path = os.path.join("uploads", file.filename)
    file.save(save_path)
    print(f"Received file: {save_path}")

    #  使用whisper处理音频文件
    raw_text = transcribe_audio(save_path)

    summary_text = summarize_class_content(raw_text)
    summary_text = remove_markdown(summary_text)

    # 模拟返回内容
    result = {"message": summary_text, "filename": file.filename}

    # 设置响应头并返回
    response = make_response(jsonify(result))
    return response, 200


@app.route('/test', methods=['POST'])
def test_api():
    # 接收 POST 请求中的 JSON 数据
    data = request.json

    # 检查数据是否包含需要的键
    if not data or 'message' not in data:
        return jsonify({"error": "Missing 'message' in request"}), 400

    # 返回测试响应
    response = {
        "received_message": data["message"],
        "status": "Success",
        "message_length": len(data["message"])
    }
    return jsonify(response)


def recognize_audio(file_path):
    # 模拟语音识别逻辑
    return "audio recognize result"


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
