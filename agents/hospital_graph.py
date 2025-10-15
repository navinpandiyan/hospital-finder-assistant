from langgraph.graph import StateGraph, END
from db.models import HospitalFinderState
from agents.agent_tools import transcribe_audio_tool, recognize_query_tool, text_to_speech_tool, hospital_lookup_tool
import asyncio
from config.config import LOGGER, MAX_TURNS
from utils.utils import play_audio, record_audio, summarize_conversation

graph = StateGraph(HospitalFinderState)

# ----------------------------
# Node 1: Record → Transcribe → Recognize
# ----------------------------
async def record_transcribe_recognize(state: HospitalFinderState):
    is_clarify_turn = state.turn_count > 0
    audio_filename = f"clarify_{state.uid}.wav" if is_clarify_turn else f"{state.uid}.wav"
    audio_path = await record_audio(output_filename=f"audios/input/{audio_filename}", duration=5)

    # Store in clarify vs original fields
    if is_clarify_turn:
        state.clarify_user_response_audio_path = audio_path
    else:
        state.input_audio_path = audio_path

    # --- Transcribe ---
    transcription_result = await transcribe_audio_tool.ainvoke({
        "audio_path": audio_path,
        "uid": state.uid
    })

    # --- Recognize query ---
    recognition_result = await recognize_query_tool.ainvoke({
        "query_text": transcription_result.get("transcribed_text"),
        "uid": state.uid,
        "use_llm": False
    })
    # LOGGER.info(f"Recognition result before merge: {recognition_result}")

    # Merge or assign clarify fields
    if is_clarify_turn:
        state.clarify_transcription = transcription_result
        state.clarify_recognition = recognition_result

        # Merge new data into main recognition (location, hospital type, insurance)
        if not state.recognition.get("location") and recognition_result.get("location"):
            state.recognition["location"] = recognition_result["location"]
            state.recognition["location_coordinates"] = recognition_result.get("location_coordinates")

        state.recognition["hospital_type"] = list(
            set(state.recognition.get("hospital_type", []) + recognition_result.get("hospital_type", []))
        )
        state.recognition["insurance"] = list(
            set(state.recognition.get("insurance", []) + recognition_result.get("insurance", []))
        )
    else:
        state.transcription = transcription_result
        state.recognition = recognition_result

    state.turn_count += 1
    return state

graph.add_node("record_transcribe_recognize", record_transcribe_recognize)

# ----------------------------
# Node 2: Clarifier (ask for missing location)
# ----------------------------
async def clarifier(state: HospitalFinderState):
    location = (state.recognition or {}).get("location")
    if location:
        return state  # location already found

    if state.turn_count >= MAX_TURNS:
        LOGGER.warning("Max turns reached without location.")
        state.final_response = {"error": "Location not provided after multiple attempts."}
        return state

    question_text = "I didn't catch your location. Could you please tell me where you are located?"
    LOGGER.info(f"Clarify for Missing Location: {question_text}")
    tts_result = await text_to_speech_tool.ainvoke({
        "text": question_text,
        "uid": state.uid
    })
    state.clarify_bot_response_audio_path = tts_result["audio_path"]
    await play_audio(tts_result["audio_path"])
    return state

graph.add_node("clarifier", clarifier)

# ----------------------------
# Node 3: Find Hospitals
# ----------------------------
async def find_hospitals(state: HospitalFinderState):
    if not state.recognition or not state.recognition.get("location_coordinates"):
        state.final_response = {"error": "Location coordinates missing."}
        return state

    user_lat, user_lon = state.recognition["location_coordinates"]
    hospitals = await hospital_lookup_tool.ainvoke({
        "user_lat": user_lat,
        "user_lon": user_lon,
        "hospital_types": state.recognition.get("hospital_type"),
        "insurance_providers": state.recognition.get("insurance")
    })
    state.hospitals_found = hospitals
    return state

graph.add_node("find_hospitals", find_hospitals)

# ----------------------------
# Node 4: Generate Response
# ----------------------------
async def generate_response(state: HospitalFinderState):
    if not state.hospitals_found:
        response_text = "I couldn't find any hospitals matching your criteria."
    else:
        hospital_list = "\n".join([f"- {h['hospital_name']} ({h['distance_km']:.2f} km away)" for h in state.hospitals_found])
        response_text = f"Hospitals near you:\n{hospital_list}"

    tts_result = await text_to_speech_tool.ainvoke({
        "text": response_text,
        "uid": state.uid,
        "convert_to_dialogue": False
    })

    await play_audio(tts_result["audio_path"])
    state.final_response = tts_result
    state.final_response_audio_path = tts_result["audio_path"]
    
    await summarize_conversation(state)
    return state

graph.add_node("generate_response", generate_response)

# ----------------------------
# Conditional edges to loop clarifier
# ----------------------------
def location_check(state: HospitalFinderState):
    if state.recognition and state.recognition.get("location"):
        return "location_found"
    if state.turn_count >= MAX_TURNS:
        return "max_turns_reached"
    return "location_missing"

graph.add_edge("record_transcribe_recognize", "clarifier")
graph.add_conditional_edges(
    "clarifier",
    location_check,
    {
        "location_missing": "record_transcribe_recognize",  # loop until location
        "location_found": "find_hospitals",
        "max_turns_reached": "generate_response"
    }
)

graph.add_edge("find_hospitals", "generate_response")
graph.add_edge("generate_response", END)

# ----------------------------
# Entry point
# ----------------------------
graph.set_entry_point("record_transcribe_recognize")

hospital_finder_graph = graph.compile()
