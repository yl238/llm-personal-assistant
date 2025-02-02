import os

from dotenv import load_dotenv
from flask import Flask, jsonify, request

from llm_processing import LLMProcessor
from transcribe_audio import AudioTranscriber
from youtube_evaluator import YouTubeVideoEvaluator

app = Flask(__name__)

# Instantiate the evaluator classes.
audio_transcriber = (
    AudioTranscriber()
)  # Uses WHISPER_PATH from environment if not passed explicitly.
llm_processor = LLMProcessor()  # Uses OPENAI_API_KEY from environment.
video_evaluator = YouTubeVideoEvaluator(audio_transcriber, llm_processor)


@app.route("/evaluate", methods=["POST"])
def evaluate():
    """
    Expects a JSON POST with {"url": "YouTube video URL"}.
    Returns a JSON response with the summary.
    """
    data = request.get_json()
    if not data or "url" not in data:
        return jsonify({"error": "Missing 'url' in request body."}), 400

    youtube_url = data["url"]
    try:
        summary = video_evaluator.evaluate_video(youtube_url)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"summary": summary})


# ------------------------------------------------------------------------------
# Run the Flask app
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
