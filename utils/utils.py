import json
import pyaudio
import asyncio
import wave
import os
from settings.config import LOGGER
from db.models import HospitalFinderState
from geopy.geocoders import Nominatim

def get_lat_long(location_name):
    """
    Retrieves the latitude and longitude for a given location name.

    Args:
        location_name (str): The name of the location (e.g., "Eiffel Tower, Paris").

    Returns:
        tuple: A tuple containing (latitude, longitude) if found, otherwise None.
    """
    geolocator = Nominatim(user_agent="HospitalsSuggestor")
    try:
        location = geolocator.geocode(location_name)
        if location:
            return (location.latitude, location.longitude)
        else:
            return None
    except Exception as e:
        LOGGER.debug(f"Error during geocoding: {e}")
        return None

# Example usage
location_to_find = "Times Square, New York"
coordinates = get_lat_long(location_to_find)

if coordinates:
    latitude, longitude = coordinates
    LOGGER.debug(f"Latitude for '{location_to_find}': {latitude}")
    LOGGER.debug(f"Longitude for '{location_to_find}': {longitude}")
else:
    LOGGER.debug(f"Could not find coordinates for '{location_to_find}'.")

async def record_audio(output_filename="input_audio.wav", duration=5, rate=44100, chunk=1024, channels=1):
    """
    Records audio from the microphone and saves it to a WAV file.

    Args:
        output_filename (str): The name of the output WAV file.
        duration (int): The duration of the recording in seconds.
        rate (int): The sample rate (samples per second).
        chunk (int): The number of frames per buffer.
        channels (int): The number of audio channels (1 for mono, 2 for stereo).
    """
    p = pyaudio.PyAudio()

    stream = p.open(format=pyaudio.paInt16,
                    channels=channels,
                    rate=rate,
                    input=True,
                    frames_per_buffer=chunk)

    LOGGER.info("🎙️ Please speak your query now. Recording for 5 seconds...")

    frames = []
    for _ in range(0, int(rate / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)

    LOGGER.debug("Recording finished.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    # Ensure the directory exists
    output_dir = os.path.dirname(output_filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    wf = wave.open(output_filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close()

    LOGGER.debug(f"Audio saved to {output_filename}")
    return output_filename

async def play_audio(audio_path, chunk=1024):
    """
    Plays a WAV audio file.

    Args:
        audio_path (str): The path to the WAV audio file.
        chunk (int): The number of frames per buffer.
    """
    if not os.path.exists(audio_path):
        LOGGER.debug(f"Error: Audio file not found at {audio_path}")
        return

    wf = wave.open(audio_path, 'rb')
    p = pyaudio.PyAudio()

    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    data = wf.readframes(chunk)
    while data:
        stream.write(data)
        data = wf.readframes(chunk)

    stream.stop_stream()
    stream.close()
    p.terminate()
    LOGGER.debug(f"Finished playing {audio_path}")

# You can add other utility functions here if needed

async def summarize_conversation(final_state: HospitalFinderState):
    print("\n===== Conversation Summary =====\n")
    print(f"User ID: {final_state.uid}")
    print(f"Total Turns: {final_state.turn_count}\n")
    
    # Initial transcription and recognition
    transcription = final_state.transcription or {}
    recognition = final_state.recognition or {}
    print("---- Initial Query ----")
    print(f"Transcribed Text: {transcription.get('transcribed_text', 'N/A')}")
    print(f"Recognized Intent: {recognition.get('intent', 'N/A')}")
    print(f"Recognized Location: {recognition.get('location', 'N/A')}")
    print(f"Hospital Types: {', '.join(recognition.get('hospital_type', [])) or 'Any'}")
    print(f"Insurance: {', '.join(recognition.get('insurance', [])) or 'Any'}\n")
    
    # Clarification (if any)
    if final_state.clarify_transcription or final_state.clarify_recognition:
        clarify_trans = final_state.clarify_transcription or {}
        clarify_recog = final_state.clarify_recognition or {}
        print("---- Clarification ----")
        print(f"Clarify Transcribed Text: {clarify_trans.get('transcribed_text', 'N/A')}")
        print(f"Clarify Recognized Location: {clarify_recog.get('location', 'N/A')}")
        print(f"Clarify Hospital Types: {', '.join(clarify_recog.get('hospital_type', [])) or 'Any'}")
        print(f"Clarify Insurance: {', '.join(clarify_recog.get('insurance', [])) or 'Any'}\n")
    
    # Hospitals found
    hospitals = final_state.hospitals_found or []
    print(f"---- Hospitals Found ({len(hospitals)}) ----")
    for h in hospitals:
        print(f"- {h['hospital_name']} ({h['distance_km']:.2f} km away)")
        print(f"  Location: {h['location']}")
        print(f"  Types: {', '.join(h.get('hospital_type', []))}")
        print(f"  Insurance: {', '.join(h.get('insurance_providers', []))}\n")
    
    # Final response
    final_response = final_state.final_response or {}
    print("---- Bot Response ----")
    # print(f"Text: {final_response.get('text', 'N/A')}")
    print(f"Response: {final_response.get('dialogue', 'N/A')}")
    print(f"Audio Path: {final_response.get('audio_path', final_state.final_response_audio_path)}\n")

async def save_state(state: HospitalFinderState, output_dir="outputs"):
    """
    Save the current HospitalFinderState as a JSON file in outputs/<uid>.json
    """
    # Ensure the directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Dump state to dictionary
    state_dict = state.model_dump()
    
    # File path
    file_path = os.path.join(output_dir, f"{state.uid}.json")
    
    # Write JSON asynchronously using a thread
    await asyncio.to_thread(
        lambda: json.dump(state_dict, open(file_path, "w", encoding="utf-8"), indent=4)
    )
    
    return file_path