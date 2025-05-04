import streamlit as st
from youtube_evaluator import YouTubeVideoEvaluator
from llm_processing import LLMProcessor
from transcribe_audio import AudioTranscriber

st.set_page_config(page_title="YouTube Video Evaluator", layout="centered")

st.title("YouTube Video Evaluator")
st.write("Paste a YouTube URL below to get a summary of the video.")

youtube_url = st.text_input("YouTube URL", "")

if "evaluator" not in st.session_state:
    audio_transcriber = AudioTranscriber()
    llm_processor = LLMProcessor()
    st.session_state.evaluator = YouTubeVideoEvaluator(audio_transcriber, llm_processor)

if st.button("Evaluate") and youtube_url.strip():
    with st.spinner("Processing... This may take a while for long videos."):
        try:
            summary = st.session_state.evaluator.evaluate_video(youtube_url.strip())
            st.subheader("Summary")
            st.write(summary)
        except Exception as e:
            st.error(f"Error: {e}")