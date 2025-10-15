import os
import uuid
import asyncio
from typing import Optional
from settings.config import LOGGER, TEXT_TO_DIALOGUE_MODEL, TEXT_TO_DIALOGUE_TEMPERATURE
from settings.client import openai_client, async_llm_client
from db.models import TTSResponseModel
from settings.prompts import TEXT_TO_DIALOGUE_SYSTEM_PROMPT, TEXT_TO_DIALOGUE_USER_PROMPT


async def text_to_speech_wrapper(text: str, uid: str, output_dir: str = "audios/output", convert_to_dialogue: bool = False) -> dict:
    """
    Converts text to speech using OpenAI's TTS API and saves the audio to a file.
    Returns a dictionary with 'uid', 'text', and 'audio_path'.
    """
    client = openai_client
    
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate a unique filename
        audio_filename = f"{uid}_{str(uuid.uuid4())}.mp3"
        audio_path = os.path.join(output_dir, audio_filename)

        # LOGGER.info(f"Generating speech for text: '{text[:50]}...'")
        # Convert text to dialogue format if needed
        if convert_to_dialogue:
            tts_response = await text_to_dialogue(text)
            
            dialogue = tts_response.dialogue if tts_response and tts_response.dialogue else text
            tone = tts_response.tone if tts_response and tts_response.tone else "neutral"
        else:
            dialogue = text
            tone = "generic"
        
        # Perform TTS using OpenAI API
        response = await asyncio.to_thread(
            client.audio.speech.create,
            model="tts-1",
            voice="alloy",
            input=dialogue,
            response_format="wav",
        )
        
        # Save the audio to a file
        await asyncio.to_thread(response.stream_to_file, audio_path)
        
        # LOGGER.info(f"Speech generated and saved to {audio_path}")
        
        return {
            "uid": uid,
            "text": text,
            "dialogue": dialogue,
            "tone": tone,
            "audio_path": audio_path,
            "llm_used": convert_to_dialogue
        }
    except Exception as e:
        LOGGER.error(f"Text-to-speech failed: {e}")
        raise

async def text_to_dialogue(text: str) -> str:
    """
    Converts text to a dialogue-friendly format.
    This is a placeholder for any text processing needed before TTS.
    """
    user_message = TEXT_TO_DIALOGUE_USER_PROMPT.format(text=text)
    
    response = await async_llm_client.beta.chat.completions.parse(
        model=TEXT_TO_DIALOGUE_MODEL,
        messages=[
            {"role": "system", "content": TEXT_TO_DIALOGUE_SYSTEM_PROMPT},
            {"role": "user", "content": user_message}],
        temperature=TEXT_TO_DIALOGUE_TEMPERATURE,
        response_format=TTSResponseModel
    )

    llm_output = response.choices[0].message.parsed

    # The parsed output is already available as a Pydantic model
    llm_output: TTSResponseModel = response.choices[0].message.parsed

    output_dialogue = llm_output.dialogue if llm_output.dialogue else text
    output_tone = llm_output.tone if llm_output.tone else "neutral"

    # Return the Pydantic model (or dict if needed)
    return TTSResponseModel(dialogue=output_dialogue, tone=output_tone)
