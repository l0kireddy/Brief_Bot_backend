from flask import Flask, request, jsonify
from flask_cors import CORS
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai import Credentials
import whisper
import os
import re
import mimetypes
import ffmpeg
from datetime import datetime
from dotenv import load_dotenv  # ✅ dotenv added

# ✅ Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# 🎤 Load Whisper model only once
whisper_model = whisper.load_model("base")

# 🔐 Load IBM credentials from environment
api_key = os.getenv("API_KEY")
project_id = os.getenv("PROJECT_ID")
region = os.getenv("REGION", "us-south")

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

# 📁 Ensure temp directory exists
os.makedirs("temp", exist_ok=True)

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    filename = datetime.now().strftime("file_%Y%m%d_%H%M%S")
    ext = os.path.splitext(file.filename)[-1]
    video_path = f"temp/{filename}{ext}"
    file.save(video_path)

    audio_path = f"temp/{filename}.mp3"

    try:
        # 🎞️ Extract audio from video or use audio directly
        mimetype = mimetypes.guess_type(video_path)[0]
        if mimetype and mimetype.startswith("video"):
            (
                ffmpeg
                .input(video_path)
                .output(audio_path, format='mp3', acodec='libmp3lame')
                .run(overwrite_output=True)
            )
        else:
            audio_path = video_path

        # 🧠 Transcribe using Whisper
        result = whisper_model.transcribe(audio_path)
        transcript = result["text"]

        # ✍️ Prompt Granite to summarize
        prompt = f"""
<think>
You are a smart assistant. Given this meeting transcript, summarize the key points and action items with deadlines and owners.

Transcript:
{transcript}
</think>
<response>
"""
        summary = granite_model.generate_text(prompt)
        clean_summary = re.sub(r"[#*`>-]", "", summary).strip()

        return jsonify({
            'transcript': transcript,
            'summary': clean_summary
        })

    finally:
        # 🧹 Cleanup
        if os.path.exists(video_path):
            os.remove(video_path)
        if audio_path != video_path and os.path.exists(audio_path):
            os.remove(audio_path)

@app.route("/")
def home():
    return "✅ Flask backend is running."

if __name__ == '__main__':
    app.run(debug=True)
