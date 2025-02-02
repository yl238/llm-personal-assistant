import os

from dotenv import load_dotenv
from openai import OpenAI


class LLMProcessor:
    def __init__(self, model="gpt-4o", prompt_only=False):
        """
        Initializes the LLM processor.

        Parameters:
          - model: the OpenAI model to use.
        """
        load_dotenv()
        self.model = model
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not set in environment.")
        self.client = OpenAI(api_key=self.api_key)
        self.prompt_only = prompt_only

    def summarise_transcript(self, transcript: str) -> str:
        """
        Uses the LLM to summarise the transcript and extract main points with timestamps.

        Parameters:
          - transcript: the full transcript text.

        Returns:
          - A summary text.
        """
        prompt = f"""System Message

You are an expert summarizer. Your goal is to:

1. Read and analyze the user-provided video transcript without adding external knowledge or assumptions.
2. Provide an overall summary of the video’s content in a concise form.
3. List key moments from the video, including timestamps and a brief explanation of each.
4. Highlight any novel or unique ideas presented in the video.
5. Correct obvious spelling mistakes in the transcript based on context, when necessary.
6. Avoid any information that cannot be directly inferred from the transcript.
Adhere strictly to the transcript. Do not include unrelated or speculative details.

Transcript to Summarize:
{transcript}

Task:
Provide:
1. A short overall summary of what the video is about.
2. Key moments with timestamps and a brief explanation for each.
3. Any novel or unique insights or ideas introduced.
- Remember to correct spelling mistakes where you’re certain of the intended word based on context.
- Do not add any details or interpretations beyond what is contained in the transcript
            """
        if self.prompt_only:
            return prompt
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=10000,
            )
            summary = response.choices[0].message.content.strip()
            return summary
