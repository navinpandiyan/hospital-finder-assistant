import uuid
from langgraph.graph import StateGraph, END
from db.models import HospitalFinderState
from graphs.graph_tools import (
    hospital_lookup_rag_tool,
    transcribe_audio_tool,
    recognize_query_tool,
    text_to_speech_tool,
    hospital_lookup_tool
)
from settings.config import (
    DEFAULT_DISTANCE_KM,
    DEFAULT_N_HOSPITALS_TO_RETURN,
    LOGGER,
    MAX_TURNS,
    MODE,
    TEXT_TO_DIALOGUE,
    USE_LLM_FOR_RECOGNITION,
    LOOKUP_MODE
)
from utils.utils import play_audio, save_state, summarize_conversation
from tools.record import record_audio

graph = StateGraph(HospitalFinderState, config={"recursion_limit": 50})

# ----------------------------
# Node: Handle user input (record or chatbot text)
# ----------------------------
async def handle_user_input(state: HospitalFinderState):
    is_chatbot = MODE == "chatbot"
    is_clarify_turn = state.turn_count > 0

    if state.uid is None:
        state.uid = str(uuid.uuid4())

    # --- Get user input ---
    if is_chatbot:        
        transcription_text = input("You: ").strip().lower()
        transcription_result = {"transcribed_text": transcription_text}
        audio_path = None
    else:
        # Voice mode
        audio_filename = f"clarify_{state.uid}.wav" if is_clarify_turn else f"{state.uid}.wav"
        LOGGER.info("Recording... Please speak now.")
        
        audio_path = await record_audio(output_filename=f"audios/input/{audio_filename}")
        transcription_result = await transcribe_audio_tool.ainvoke({
            "audio_path": audio_path,
            "uid": state.uid
        })
        transcription_text = transcription_result.get("transcribed_text", "").lower()
        print(f"You: {transcription_text}")
    # --- Recognize query ---
    recognition_result = await recognize_query_tool.ainvoke({
        "query_text": transcription_text,
        "uid": state.uid,
        "use_llm": USE_LLM_FOR_RECOGNITION
    })

    if recognition_result.get("intent", "").lower() == "exit":
        state.user_wants_exit = True
        return state

    # --- Assign fields ---
    if is_clarify_turn:
        if not is_chatbot:
            state.clarify_user_response_audio_path = audio_path
            state.clarify_transcription = transcription_result
        else:
            state.clarify_transcription = transcription_result

        state.clarify_recognition = recognition_result
        if not state.recognition.get("location") and recognition_result.get("location"):
            state.recognition["location"] = recognition_result["location"]
            state.recognition["location_coordinates"] = recognition_result.get("location_coordinates")
    else:
        if not is_chatbot:
            state.input_audio_path = audio_path
            state.transcription = transcription_result
        else:
            state.transcription = transcription_result

        state.recognition = recognition_result
        state.clarify_recognition = {}
        state.clarify_transcription = {}
        state.clarify_bot_response_audio_path = None
        if not is_chatbot:
            state.clarify_user_response_audio_path = None

    state.turn_count += 1
    return state

graph.add_node("handle_user_input", handle_user_input)

# ----------------------------
# Node: Clarifier
# ----------------------------
async def clarifier(state: HospitalFinderState):
    intent = (state.recognition or {}).get("intent")
    location = (state.recognition or {}).get("location")
    hospital_names = (state.recognition or {}).get("hospital_names", [])

    needs_clarification = False
    if intent in ["find_nearest", "find_best", "find_by_insurance"]:
        needs_clarification = not location
    elif intent == "compare_hospitals":
        needs_clarification = len(hospital_names) < 2

    if not needs_clarification:
        return state

    if state.turn_count >= MAX_TURNS:
        LOGGER.warning("Max turns reached without required info.")
        print("Bot: Max turns reached without required info. Exiting!")
        state.final_response = {"error": "Required information not provided after multiple attempts."}
        state.user_wants_exit = True
        return state

    # --- Build prompt ---
    parts = []
    hospital_types = state.recognition.get("hospital_type", [])
    insurance_providers = state.recognition.get("insurance", [])

    if intent in ["find_nearest", "find_best", "find_by_insurance"]:
        parts.append("I didn't catch your location.")
        parts.append("Could you please tell me the city or area you're in?")
    elif intent == "compare_hospitals":
        parts.append("I didn't catch which hospitals you want to compare.")
        parts.append("Could you please provide at least two hospital names?")

    question_text = " ".join(parts)

    print(f"Bot: {question_text}")
    if MODE == "voicebot":
        tts_result = await text_to_speech_tool.ainvoke({
            "text": question_text,
            "uid": state.uid,
            "convert_to_dialogue": TEXT_TO_DIALOGUE
        })
        state.clarify_bot_response_audio_path = tts_result["audio_path"]
        LOGGER.info(f'BOT: {tts_result["dialogue"]}')
        await play_audio(tts_result["audio_path"])

    return state

graph.add_node("clarifier", clarifier)

# ----------------------------
# Node: Find Hospitals
# ----------------------------
async def find_hospitals(state: HospitalFinderState):
    if not state.recognition or not state.recognition.get("location_coordinates"):
        state.final_response = {"error": "Location coordinates missing."}
        return state

    user_loc = state.recognition["location"]
    user_query = state.recognition.get("output_query")
    user_lat, user_lon = state.recognition["location_coordinates"]
    n_hospitals = state.recognition.get("n_hospitals", DEFAULT_N_HOSPITALS_TO_RETURN)
    distance_km = state.recognition.get("distance_km", DEFAULT_DISTANCE_KM)

    if LOOKUP_MODE == "simple":
        selected_hospitals = await hospital_lookup_tool.ainvoke({
            "user_lat": user_lat,
            "user_lon": user_lon,
            "intent": state.recognition.get("intent", "find_nearest"),
            "hospital_types": state.recognition.get("hospital_type"),
            "insurance_providers": state.recognition.get("insurance"),
            "n_hospitals": n_hospitals,
            "distance_km_radius": distance_km
        })
        if not selected_hospitals:
            response_text = "I couldn't find any hospitals matching your criteria."
        else:
            hospital_list = "\n".join([f"- {h['hospital_name']} ({h['distance_km']:.2f} km away)" 
                                        for h in selected_hospitals])
            response_text = f"Hospitals near you:\n{hospital_list}"
        retrieved_hospitals = selected_hospitals
    elif LOOKUP_MODE == "rag":
        retrieved_hospitals, selected_hospitals, response_text = await hospital_lookup_rag_tool.ainvoke({
            "user_query": user_query,
            "user_loc": user_loc,
            "user_lat": user_lat,
            "user_lon": user_lon,
            "intent": state.recognition.get("intent", "find_nearest"),
            "hospital_types": state.recognition.get("hospital_type"),
            "hospital_names": state.recognition.get("hospital_names"),
            "insurance_providers": state.recognition.get("insurance"),
            "n_hospitals": n_hospitals,
            "distance_km_radius": distance_km,
            # "extra_results": 5
        })

    state.hospitals_found = {"retrieved": retrieved_hospitals, "selected": selected_hospitals}
    state.final_response_text = response_text
    return state

graph.add_node("find_hospitals", find_hospitals)

# ----------------------------
# Node: Generate Response
# ----------------------------
async def generate_response(state: HospitalFinderState):
    response_text = state.final_response_text or "I couldn't find any hospitals matching your criteria."
    print(f"Bot: {response_text}")
    if MODE == "voicebot":
        tts_result = await text_to_speech_tool.ainvoke({
            "text": response_text,
            "uid": state.uid,
            "convert_to_dialogue": TEXT_TO_DIALOGUE
        })
        await play_audio(tts_result["audio_path"])
        state.final_response_audio_path = tts_result["audio_path"]
        # await summarize_conversation(state)
    
    await save_state(state)

    followup_text = "Do you want help with any other query?"
    print(f"Bot: {followup_text}")
    if MODE == "voicebot":
        followup_tts = await text_to_speech_tool.ainvoke({
            "text": followup_text,
            "uid": state.uid,
            "convert_to_dialogue": TEXT_TO_DIALOGUE
        })
        await play_audio(followup_tts["audio_path"])

    # Reset state except UID
    state = HospitalFinderState()
    return state

graph.add_node("generate_response", generate_response)

# ----------------------------
# Normal edges
# ----------------------------
graph.add_edge("find_hospitals", "generate_response")
graph.add_edge("generate_response", "handle_user_input")

# ----------------------------
# Conditional edges
# ----------------------------
def clarifier_conditional(state: HospitalFinderState):
    if getattr(state, "user_wants_exit", False):
        return "end_conversation"
    intent = state.recognition.get("intent")
    location = state.recognition.get("location")
    hospital_names = state.recognition.get("hospital_names", [])

    if intent in ["find_nearest", "find_best", "find_by_insurance"]:
        if not location:
            return "location_missing" if state.turn_count < MAX_TURNS else "max_turns_reached"
        return "location_found"
    elif intent == "compare_hospitals":
        if len(hospital_names) < 2:
            return "hospital_names_missing" if state.turn_count < MAX_TURNS else "max_turns_reached"
        return "hospital_names_found"
    elif intent == "find_by_hospital":
        return "hospital_names_found"
    return "needs_clarification"

# Conditional edge from first node to clarifier or END
def record_conditional(state: HospitalFinderState):
    if getattr(state, "user_wants_exit", False):
        return "end_conversation"
    return "clarifier"

graph.add_conditional_edges(
    "handle_user_input",
    record_conditional,
    {
        "clarifier": "clarifier",
        "end_conversation": END
    }
)

graph.add_conditional_edges(
    "clarifier",
    clarifier_conditional,
    {
        "location_missing": "handle_user_input",
        "location_found": "find_hospitals",
        "hospital_names_missing": "handle_user_input",
        "hospital_names_found": "find_hospitals",
        "max_turns_reached": "generate_response",
        "end_conversation": END,
        "needs_clarification": "handle_user_input"
    }
)

# Entry point
graph.set_entry_point("handle_user_input")
hospital_finder_graph = graph.compile()
