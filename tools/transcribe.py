"""
tools/transcribe.py
-------------------
Audio Transcription Module – Uses OpenAI Whisper API

Objective:
-----------
Convert recorded audio files (e.g., .wav, .mp3) into transcribed text
for downstream NLP or LLM-based intent recognition.
"""

import os
import asyncio
from settings.config import LOGGER, TRANSCRIBER_LANGUAGE, TRANSCRIBER_OPENAI_MODEL
from settings.client import openai_client

async def transcribe_wrapper(audio_path: str, uid: str) -> dict:
    """
    Transcribes an audio file using OpenAI Whisper API.
    
    Args:
        audio_path (str): Path to the recorded audio file.
        uid (str): Unique session identifier.

    Returns:
        dict: {
            "uid": str,
            "audio_path": str,
            "transcribed_text": str
        }
    """
    client = openai_client

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # LOGGER.info(f"🎙️ Transcribing audio for UID={uid}: {audio_path}")

    try:
        # Open file safely and stream to OpenAI’s transcription model
        with open(audio_path, "rb") as audio_file:
            response = await asyncio.to_thread(
                client.audio.transcriptions.create,
                model=TRANSCRIBER_OPENAI_MODEL,
                file=audio_file,
                language=TRANSCRIBER_LANGUAGE
            )

        transcribed_text = response.text.strip() if hasattr(response, "text") else ""

        LOGGER.info(f"📝 USER: '{transcribed_text}'")

        return {
            "uid": uid,
            "audio_path": audio_path,
            "transcribed_text": transcribed_text,
        }

    except Exception as e:
        LOGGER.error(f"Transcription failed for {uid}: {e}")
        raise
