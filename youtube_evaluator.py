import os
import subprocess
import tempfile
from urllib.parse import parse_qs, urlparse

from dotenv import load_dotenv
from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi

from llm_processing import LLMProcessor
from transcribe_audio import AudioTranscriber


class YouTubeVideoEvaluator:
    def __init__(
        self, audio_transcriber: AudioTranscriber, llm_processor: LLMProcessor
    ):
        self.audio_transcriber = audio_transcriber
        self.llm_processor = llm_processor

    def evaluate_video(self, youtube_url: str) -> str:
        """
        Evaluates a YouTube video by obtaining (or generating) its transcript and then summarising it.

        Parameters:
          - youtube_url: URL of the YouTube video.

        Returns:
          - A summary text that highlights main points and timestamps.
        """
        video_id = self.extract_video_id(youtube_url)
        transcript_text = ""
        try:
            # Try to fetch transcript via the API.
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            transcript_text = "\n".join(
                [
                    f"[{self.format_timestamp(item['start'])}] {item['text']}"
                    for item in transcript_list
                ]
            )
        except Exception as e:
            print(f"No transcript available via API: {e}")

        if not transcript_text.strip():
            print(
                "Transcript not found. Downloading video and transcribing audio..."
            )
            yt = YouTube(youtube_url)
            # Prefer an audio-only stream if available.
            stream = yt.streams.filter(only_audio=True).first()
            if not stream:
                stream = yt.streams.first()
            temp_dir = tempfile.gettempdir()
            filename = f"{video_id}.{stream.subtype}"
            download_path = os.path.join(temp_dir, filename)
            print(f"Downloading to {download_path} ...")
            stream.download(output_path=temp_dir, filename=filename)
            print("Download complete.")
            # Transcribe the downloaded file. (Use inplace=False to preserve the original.)
            transcript_file = self.audio_transcriber.transcribe(
                download_path, inplace=False
            )
            with open(transcript_file, "r", encoding="utf-8") as f:
                transcript_text = f.read()
            # Optionally, remove temporary files:
            os.remove(download_path)
            # os.remove(transcript_file)

        # Use the LLM to summarise the transcript and extract main points.
        summary = self.llm_processor.summarise_transcript(transcript_text)
        return summary

    @staticmethod
    def extract_video_id(url: str) -> str:
        """
        Extracts the video ID from a YouTube URL.
        """
        parsed = urlparse(url)
        qs = parse_qs(parsed.query)
        if "v" in qs:
            return qs["v"][0]
        # Fallback: use the last part of the path.
        return os.path.basename(parsed.path)

    @staticmethod
    def format_timestamp(seconds: float) -> str:
        """
        Formats a number of seconds as HH:MM:SS.
        """
        seconds = int(seconds)
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        return f"{h:02d}:{m:02d}:{s:02d}"


if __name__ == "__main__":
    # Create instances of our helper classes.
    audio_transcriber = (
        AudioTranscriber()
    )  # uses WHISPER_PATH from env if not passed explicitly
    
    youtube_url = input("Enter a YouTube video URL: ").strip()
    prompt_only = input("Only create LLM prompt? (y/n): ").lower().strip() == 'y'

    llm_processor = LLMProcessor(prompt_only=prompt_only)  # uses OPENAI_API_KEY from env
    evaluator = YouTubeVideoEvaluator(audio_transcriber, llm_processor)

    output = evaluator.evaluate_video(youtube_url)
    if prompt_only:
        with open('prompt.txt', 'w') as f:
            f.write(output)
    else:
        with open('video_summary.md', 'w') as f:
            f.write(output)
