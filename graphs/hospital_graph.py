import uuid
from langgraph.graph import StateGraph, END
from db.models import HospitalFinderState
from graphs.graph_tools import transcribe_audio_tool, recognize_query_tool, text_to_speech_tool, hospital_lookup_tool
from settings.config import LOGGER, MAX_TURNS
from utils.utils import play_audio, record_audio, save_state, summarize_conversation

graph = StateGraph(HospitalFinderState)

EXIT_KEYWORDS = [
    "no", "nope", "stop", "exit", "quit", "end", 
    "that's all", "done", "nothing", "bye", "goodbye"
]

# ----------------------------
# Node 1: Record → Transcribe → Recognize
# ----------------------------
async def record_transcribe_recognize(state: HospitalFinderState):
    is_clarify_turn = state.turn_count > 0

    # --- Keep same UID for entire session ---
    if state.uid is None:
        state.uid = str(uuid.uuid4())

    audio_filename = f"clarify_{state.uid}.wav" if is_clarify_turn else f"{state.uid}.wav"
    audio_path = await record_audio(output_filename=f"audios/input/{audio_filename}", duration=5)

    # --- Transcribe ---
    transcription_result = await transcribe_audio_tool.ainvoke({
        "audio_path": audio_path,
        "uid": state.uid
    })
    transcription_text = transcription_result.get("transcribed_text", "").lower()

    # --- Check for exit words ---
    if any(word in transcription_text for word in EXIT_KEYWORDS):
        state.user_wants_exit = True  # Flag to let conditional edge handle END
        return state

    # --- Recognize query ---
    recognition_result = await recognize_query_tool.ainvoke({
        "query_text": transcription_text,
        "uid": state.uid,
        "use_llm": False
    })

    # Assign or merge fields
    if is_clarify_turn:
        state.clarify_user_response_audio_path = audio_path
        state.clarify_transcription = transcription_result
        state.clarify_recognition = recognition_result

        # Merge missing fields
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
        # First turn: populate state without replacing UID
        state.input_audio_path = audio_path
        state.transcription = transcription_result
        state.recognition = recognition_result
        state.clarify_recognition = {}
        state.clarify_transcription = {}
        state.clarify_bot_response_audio_path = None
        state.clarify_user_response_audio_path = None

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
        state.user_wants_exit = True
        return state

    def build_clarifier_prompt(state: HospitalFinderState) -> str:
        hospital_types = state.recognition.get("hospital_type", [])
        insurance_providers = state.recognition.get("insurance", [])

        parts = ["I didn't catch your location."]
        if hospital_types:
            parts.append(f"You mentioned looking for {' and '.join(hospital_types)} hospitals.")
        if insurance_providers:
            parts.append(f"You also mentioned insurance providers: {' and '.join(insurance_providers)}.")

        parts.append("Could you please tell me the city or area you're in? You can also update hospital types or insurance if needed.")
        
        return " ".join(parts)
    
    question_text = build_clarifier_prompt(state)
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
        "insurance_providers": state.recognition.get("insurance"),
        "limit": 2
    })
    state.hospitals_found = hospitals
    return state

graph.add_node("find_hospitals", find_hospitals)

# ----------------------------
# Node 4: Generate Response + Ask Another Query
# ----------------------------
async def generate_response(state: HospitalFinderState):
    # --- Generate hospital list response ---
    if not state.hospitals_found:
        response_text = "I couldn't find any hospitals matching your criteria."
    else:
        hospital_list = "\n".join([f"- {h['hospital_name']} ({h['distance_km']:.2f} km away)" 
                                   for h in state.hospitals_found])
        response_text = f"Hospitals near you:\n{hospital_list}"

    tts_result = await text_to_speech_tool.ainvoke({
        "text": response_text,
        "uid": state.uid,
        "convert_to_dialogue": False
    })
    await play_audio(tts_result["audio_path"])
    state.final_response = tts_result
    state.final_response_audio_path = tts_result["audio_path"]

    # --- Ask if user wants another query ---
    followup_text = "Do you want help with any other query?"
    followup_tts = await text_to_speech_tool.ainvoke({
        "text": followup_text,
        "uid": state.uid
    })
    await play_audio(followup_tts["audio_path"])

    # Summarize conversation and save state
    await summarize_conversation(state)
    await save_state(state)
    
    # Clear state for new query except UID
    state = HospitalFinderState()
    return state

graph.add_node("generate_response", generate_response)

# ----------------------------
# Conditional edges
# ----------------------------
def clarifier_conditional(state: HospitalFinderState):
    if getattr(state, "user_wants_exit", False):
        return "end_conversation"
    if state.recognition and state.recognition.get("location"):
        return "location_found"
    if state.turn_count >= MAX_TURNS:
        return "max_turns_reached"
    return "location_missing"

graph.add_conditional_edges(
    "clarifier",
    clarifier_conditional,
    {
        "location_missing": "record_transcribe_recognize",
        "location_found": "find_hospitals",
        "max_turns_reached": "generate_response",
        "end_conversation": END
    }
)

# Conditional edge from first node to clarifier or END
def record_conditional(state: HospitalFinderState):
    if getattr(state, "user_wants_exit", False):
        return "end_conversation"
    return "clarifier"

graph.add_conditional_edges(
    "record_transcribe_recognize",
    record_conditional,
    {
        "clarifier": "clarifier",
        "end_conversation": END
    }
)

# Normal edges
graph.add_edge("find_hospitals", "generate_response")
graph.add_edge("generate_response", "record_transcribe_recognize")

# ----------------------------
# Entry point
# ----------------------------
graph.set_entry_point("record_transcribe_recognize")

hospital_finder_graph = graph.compile()
