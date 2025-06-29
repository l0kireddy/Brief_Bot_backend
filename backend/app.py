from flask import Flask, request, jsonify
from flask_cors import CORS
from moviepy.editor import VideoFileClip
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai import Credentials
import whisper
import os
import mimetypes
from datetime import datetime

app = Flask(__name__)
CORS(app)

# 🧠 Load Whisper only once
whisper_model = whisper.load_model("base")

# 🔐 IBM Granite Credentials
api_key = "Fubx************************"
project_id = "your_project_id_here"
region = "us-south"

credentials = Credentials(
    api_key=api_key,
    url=f"https://{region}.ml.cloud.ibm.com"
)

granite_model = ModelInference(
    model_id="ibm/granite-3-3-8b-instruct",
    credentials=credentials,
    project_id=project_id,
    params={
        "decoding_method": "greedy",
        "max_new_tokens": 300,
        "stop_sequences": ["</response>"]
    }
)

# 📁 Create temp folder
os.makedirs("temp", exist_ok=True)


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    filename = datetime.now().strftime("file_%Y%m%d_%H%M%S")
    ext = os.path.splitext(file.filename)[-1]
    file_path = f"temp/{filename}{ext}"
    file.save(file_path)

    try:
        mimetype = mimetypes.guess_type(file_path)[0]
        if mimetype and mimetype.startswith("video"):
            video = VideoFileClip(file_path)
            audio_path = f"temp/{filename}.wav"
            video.audio.write_audiofile(audio_path)
        else:
            audio_path = file_path

        result = whisper_model.transcribe(audio_path)
        transcript = result["text"]

        prompt = f"""
<think>
You are a smart assistant. Given this meeting transcript, summarize the key points and action items with deadlines and owners.

Transcript:
{transcript}
</think>
<response>
"""
        summary = granite_model.generate_text(prompt)

        return jsonify({
            'transcript': transcript,
            'summary': summary
        })

    finally:
        os.remove(file_path)
        if file_path != audio_path and os.path.exists(audio_path):
            os.remove(audio_path)


@app.route("/")
def home():
    return "✅ Flask backend is running."

if __name__ == '__main__':
    app.run(debug=True)
