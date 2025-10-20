"""
tools/record.py
---------------
Handles recording of user voice input for the AI Hospital Voice Assistant.
This function is called in the first node of the LangGraph: record → transcribe → recognize.
"""

import asyncio
import os
import sys
import time
import pyaudio
import wave
import audioop # Import for audio processing
from settings.config import LOGGER

# No longer imported from utils, as it's defined here.

async def record_audio(
    output_filename="input_audio.wav",
    duration=10,  # Max recording duration in seconds
    rate=44100,
    chunk=1024,
    channels=1,
    silence_threshold=100,  # RMS energy below this is considered silence
    silence_duration=3.0,  # Seconds of silence to trigger stop
):
    """
    Records audio from the microphone and saves it to a WAV file.
    Recording stops if silence is detected for `silence_duration` seconds
    or after `duration` seconds (max duration).

    Args:
        output_filename (str): The name of the output WAV file.
        duration (int): The maximum duration of the recording in seconds.
        rate (int): The sample rate (samples per second).
        chunk (int): The number of frames per buffer.
        channels (int): The number of audio channels (1 for mono, 2 for stereo).
        silence_threshold (int): RMS energy below this value is considered silence.
        silence_duration (float): Number of seconds of consecutive silence to stop recording.
    """
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=channels,
                    rate=rate,
                    input=True,
                    frames_per_buffer=chunk)

    frames = []
    silent_chunks = 0
    max_silent_chunks = int(rate / chunk * silence_duration)
    max_total_chunks = int(rate / chunk * duration)

    # Cycle dots setup
    dots = ["   ", ".  ", ".. ", "..."]
    dot_index = 0
    last_dot_time = time.time()
    dot_interval = 0.5  # seconds between dot updates

    try:
        for i in range(max_total_chunks):
            data = stream.read(chunk, exception_on_overflow=False)
            frames.append(data)

            rms = audioop.rms(data, 2)
            if rms < silence_threshold:
                silent_chunks += 1
                if silent_chunks > max_silent_chunks:
                    break
            else:
                silent_chunks = 0

            # Only update dots every dot_interval seconds
            if time.time() - last_dot_time >= dot_interval:
                sys.stdout.write(f"\r🎙️ Recording{dots[dot_index]}")
                sys.stdout.flush()
                dot_index = (dot_index + 1) % len(dots)
                last_dot_time = time.time()

    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        # Clear the line after recording ends
        sys.stdout.write("\r" + " " * 50 + "\r")
        sys.stdout.flush()

    os.makedirs(os.path.dirname(output_filename) or ".", exist_ok=True)
    wf = wave.open(output_filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close()

    LOGGER.debug(f"Audio saved to {output_filename}")
    return output_filename


async def record_audio_wrapper(uid: str) -> dict:
    """
    Asynchronously records user voice and saves it as a WAV file. Recording stops on silence.

    Args:
        uid (str): Unique session identifier for tracking.

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

        # The record_audio function now handles duration and silence detection internally
        LOGGER.info(f"🎙️ Recording user query until silence is detected or max duration is reached...")

        # Run the blocking I/O in a thread-safe async way
        await asyncio.to_thread(record_audio, audio_path)

        LOGGER.info(f"✅ Audio recording complete: {audio_path}")

        return {
            "uid": uid,
            "input_audio_path": audio_path,
        }

    except Exception as e:
        LOGGER.error(f"❌ Error during audio recording for UID={uid}: {e}")
        raise
