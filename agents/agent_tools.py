import uuid
from config.config import LOGGER
from langchain.agents import tool

from typing import List, Optional, Tuple

from utils.utils import record_audio

@tool
async def transcribe_audio_tool(audio_path: str, uid: str) -> dict:
    """
    Transcribes audio from a given file path using Whisper.
    Returns a dictionary with 'uid', 'audio_path', and 'transcribed_text'.
    """
    from tools.transcribe import transcribe_wrapper
    return await transcribe_wrapper(audio_path, uid=uid)


@tool
async def recognize_query_tool(query_text: str, uid: str, use_llm: bool = True) -> dict:
    """
    Extracts structured information (location, hospital_type, insurance) from a query text.
    Returns a dictionary with 'uid', 'query', 'intent', 'location', 'hospital_type', 'insurance'.
    """
    from tools.recognize import recognize_wrapper
    return await recognize_wrapper(query_text, uid=uid, use_llm=use_llm)

@tool
async def text_to_speech_tool(text: str, uid: str, output_dir: str = "audios/output", convert_to_dialogue: bool = False) -> dict:
    """
    Converts text to speech using OpenAI's TTS API and saves the audio to a file.
    Returns a dictionary with 'uid', 'text', and 'audio_path'.
    """
    from tools.text_to_speech import text_to_speech_wrapper
    return await text_to_speech_wrapper(text, uid, output_dir, convert_to_dialogue)

@tool
async def hospital_lookup_tool(
    user_lat: float,
    user_lon: float,
    hospital_types: Optional[List[str]] = None,
    insurance_providers: Optional[List[str]] = None,
    limit: int = 5
) -> List[dict]:
    """
    Looks up hospitals in the database based on user's location, desired hospital types,
    and insurance providers. Returns a list of matching hospitals sorted by distance.
    """
    from tools.hospital_lookup import hospital_lookup_wrapper
    return await hospital_lookup_wrapper(user_lat, user_lon, hospital_types, insurance_providers, limit)
