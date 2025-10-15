"""
tools/record.py
---------------
Handles recording of user voice input for the AI Hospital Voice Assistant.
This function is called in the first node of the LangGraph: record → transcribe → recognize.
"""

import asyncio
import uuid
import os
from config.config import LOGGER
from utils.utils import record_audio

async def record_audio_wrapper(uid: str, duration: int = 5) -> dict:
    """
    Asynchronously records user voice and saves it as a WAV file.

    Args:
        uid (str): Unique session identifier for tracking.
        duration (int, optional): Duration of recording in seconds. Defaults to 5.

    Returns:
        dict: {
            "uid": str,
            "input_audio_path": str
        }
    """
    try:
        # Generate unique file name under audios/input/
        os.makedirs("audios/input", exist_ok=True)
        audio_path = f"audios/input/{uid}.wav"

        LOGGER.info(f"🎙️ Recording user query for {duration} seconds...")

        # Run the blocking I/O in a thread-safe async way
        await asyncio.to_thread(record_audio, audio_path, duration)

        LOGGER.info(f"✅ Audio recording complete: {audio_path}")

        return {
            "uid": uid,
            "input_audio_path": audio_path,
        }

    except Exception as e:
        LOGGER.error(f"❌ Error during audio recording for UID={uid}: {e}")
        raise
