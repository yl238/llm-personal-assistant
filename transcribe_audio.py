import os
import subprocess
from dotenv import load_dotenv

class AudioTranscriber:
    def __init__(self, 
                 ffmpeg_path="/opt/homebrew/bin/ffmpeg", 
                 whisper_path=None, 
                 model="ggml-medium.bin"):
        """
        Initializes the AudioTranscriber.

        Parameters:
          - ffmpeg_path: path to your ffmpeg binary.
          - whisper_path: path to your Whisper installation (or set WHISPER_PATH in your env).
          - model: name of the Whisper model to use.
        """
        load_dotenv()
        self.ffmpeg_path = ffmpeg_path
        self.whisper_path = whisper_path or os.getenv("WHISPER_PATH")
        if not self.whisper_path:
            raise ValueError("Whisper path not set. Please set WHISPER_PATH in your environment or pass it as an argument.")
        self.model = model
        self.whisper_cli = os.path.join(self.whisper_path, "build", "bin", "whisper-cli")
        self.model_path = os.path.join(self.whisper_path, "models", self.model)
        
        # Parameters for the audio conversion.
        self.audio_codec = "pcm_s16le"
        self.channels = "2"
        self.sample_rate = "16000"

    def convert_to_wav(self, input_filepath, inplace=True):
        """
        Converts the given file (audio or video) to WAV format using ffmpeg.
        If inplace is True, the WAV file replaces the original file;
        otherwise, the new file is returned.

        Parameters:
          - input_filepath: path to the input file.
          - inplace: whether to overwrite the input file with the WAV file.

        Returns:
          - The path to the WAV file.
        """
        wav_filepath = input_filepath + ".wav"
        command = [
            self.ffmpeg_path,
            "-i", input_filepath,
            "-acodec", self.audio_codec,
            "-ac", self.channels,
            "-ar", self.sample_rate,
            wav_filepath
        ]
        subprocess.run(command, check=True)
        if inplace:
            os.rename(wav_filepath, input_filepath)
            return input_filepath
        else:
            return wav_filepath

    def transcribe(self, input_filepath, language="auto", output_format="txt", inplace=True):
        """
        Transcribes the given file by first converting it to WAV (if needed) and then
        calling the Whisper CLI.

        Parameters:
          - input_filepath: path to the file (audio or video).
          - language: language parameter for Whisper.
          - output_format: transcript file format (e.g. "txt").
          - inplace: if False, the conversion will produce a separate WAV file.

        Returns:
          - The path to the transcript file.
        """
        wav_filepath = self.convert_to_wav(input_filepath, inplace=inplace)
        command = [
            self.whisper_cli,
            wav_filepath,
            "-m", self.model_path,
            f"-o{output_format}",
            "-l", language
        ]
        subprocess.run(command, check=True)
        transcript_filepath = wav_filepath + f".{output_format}"
        return transcript_filepath


# Example usage:
if __name__ == "__main__":
    # Optionally, you can pass paths here or set WHISPER_PATH in your .env file.
    transcriber = AudioTranscriber()
    transcript_file = transcriber.transcribe("path/to/your/audio/file.mp3")
    print(f"Transcript saved to: {transcript_file}")
