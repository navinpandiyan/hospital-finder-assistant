from langgraph.graph import StateGraph, END
from db.models import HospitalFinderState
from graphs.graph_tools import transcribe_audio_tool, recognize_query_tool, text_to_speech_tool, hospital_lookup_tool
from settings.config import LOGGER, MAX_TURNS
from utils.utils import play_audio
from tools.record import record_audio

graph = StateGraph(HospitalFinderState)

# ----------------------------
# Node 1: Transcribe Initial Audio
# ----------------------------
async def run_transcriber(state: HospitalFinderState):
    LOGGER.info("Please speak your query now (e.g., 'Find me a cardiology hospital in Dubai'). Recording for 5 seconds...")
    initial_audio_path = await record_audio(output_filename=f"audios/input/{state.uid}.wav")
    state.input_audio_path = initial_audio_path
    LOGGER.info(f"Transcribing initial audio from: {state.input_audio_path}")
    transcription_result = await transcribe_audio_tool.ainvoke({
        "audio_path": state.input_audio_path,
        "uid": state.uid
    })
    state.transcription = transcription_result
    LOGGER.info(f"Initial transcription: {state.transcription.get('transcribed_text')}")
    return state

graph.add_node("transcriber", run_transcriber)

# ----------------------------
# Node 2: Recognize Initial Query
# ----------------------------
async def run_recognizer(state: HospitalFinderState):
    if not state.transcription or not state.transcription.get("transcribed_text"):
        LOGGER.error("No transcription found for initial query. Cannot recognize.")
        raise ValueError("No transcription found. Cannot recognize query.")
    
    LOGGER.info(f"Recognizing initial query: {state.transcription['transcribed_text']}")
    recognition_result = await recognize_query_tool.ainvoke({
        "query_text": state.transcription["transcribed_text"],
        "uid": state.uid,
        "use_llm": False # Use LLM for initial recognition
    })
    state.recognition = recognition_result
    LOGGER.info(f"Initial recognition: {state.recognition}")
    return state

graph.add_node("recognizer", run_recognizer)

# ----------------------------
# Node 3: Ask for Missing Location
# ----------------------------
async def ask_for_location(state: HospitalFinderState):
    state.turn_count += 1
    question_text = "Which city are you looking for?"
    LOGGER.info(f"Asking for missing location: {question_text}")
    tts_result = await text_to_speech_tool.ainvoke({
        "text": question_text,
        "uid": state.uid
    })
    state.clarify_bot_response_audio_path = tts_result["audio_path"]
    state.last_question = question_text
    play_audio(tts_result["audio_path"])
    return state

graph.add_node("ask_for_location", ask_for_location)

# ----------------------------
# Node 4: Re-transcribe User Response (for follow-up)
# ----------------------------
async def re_transcribe_user_response(state: HospitalFinderState):
    LOGGER.info(f"Please respond to the bot's question. Recording for 5 seconds...")
    clarify_user_response_audio_path = await record_audio(output_filename=f"audios/input/clarify_{state.uid}.wav")
    state.clarify_user_response_audio_path = clarify_user_response_audio_path
    
    if not state.clarify_user_response_audio_path:
        LOGGER.error("No user response audio path found for re-transcription.")
        raise ValueError("No user response audio path provided.")
    
    LOGGER.info(f"Re-transcribing user response from: {state.clarify_user_response_audio_path}")
    transcription_result = await transcribe_audio_tool.ainvoke({
        "audio_path": state.clarify_user_response_audio_path,
        "uid": state.uid
    })
    state.transcription = transcription_result # Overwrite previous transcription
    LOGGER.info(f"Re-transcription: {state.transcription.get('transcribed_text')}")
    return state

graph.add_node("re_transcribe_user_response", re_transcribe_user_response)

# ----------------------------
# Node 5: Re-recognize Query (after follow-up)
# ----------------------------
async def re_recognize_query(state: HospitalFinderState):
    if not state.transcription or not state.transcription.get("transcribed_text"):
        LOGGER.error("No transcription found for re-recognition. Cannot recognize.")
        raise ValueError("No transcription found. Cannot re-recognize query.")
    
    LOGGER.info(f"Re-recognizing query: {state.transcription['transcribed_text']}")
    recognition_result = await recognize_query_tool.ainvoke({
        "query_text": state.transcription["transcribed_text"],
        "uid": state.uid,
        "use_llm": False # Use LLM for re-recognition
    })
    # Merge new recognition with existing, prioritizing new location
    if state.recognition:
        state.recognition["location"] = recognition_result.get("location") or state.recognition.get("location")
        state.recognition["location_coordinates"] = recognition_result.get("location_coordinates") or state.recognition.get("location_coordinates")
        # For hospital_type and insurance, combine unique values
        state.recognition["hospital_type"] = list(set(state.recognition.get("hospital_type", []) + recognition_result.get("hospital_type", [])))
        state.recognition["insurance"] = list(set(state.recognition.get("insurance", []) + recognition_result.get("insurance", [])))
    else:
        state.recognition = recognition_result
    LOGGER.info(f"Re-recognition (merged): {state.recognition}")
    return state

graph.add_node("re_recognize_query", re_recognize_query)

# ----------------------------
# Node 6: Find Hospitals
# ----------------------------
async def find_hospitals(state: HospitalFinderState):
    if not state.recognition or not state.recognition.get("location_coordinates"):
        LOGGER.error("No location coordinates found to find hospitals.")
        raise ValueError("Location coordinates missing for hospital lookup.")
    
    user_lat, user_lon = state.recognition["location_coordinates"]
    hospital_types = state.recognition.get("hospital_type")
    insurance_providers = state.recognition.get("insurance")

    LOGGER.info(f"Finding hospitals for lat={user_lat}, lon={user_lon}, types={hospital_types}, insurance={insurance_providers}")
    hospitals = await hospital_lookup_tool.ainvoke({
        "user_lat": user_lat,
        "user_lon": user_lon,
        "hospital_types": hospital_types,
        "insurance_providers": insurance_providers
    })
    state.hospitals_found = hospitals
    LOGGER.info(f"Found {len(hospitals)} hospitals in lookup.")
    return state

graph.add_node("find_hospitals", find_hospitals)

# ----------------------------
# Node 7: Generate Response
# ----------------------------
async def generate_response(state: HospitalFinderState):
    if not state.hospitals_found:
        response_text = "I couldn't find any hospitals matching your criteria. Please try again with different details."
        LOGGER.info("No hospitals found, generating negative response.")
    else:
        hospital_list_str = "\n".join([
            f"- {h['hospital_name']} at {h['location']} ({h['distance_km']:.2f} km away), specializing in {h['hospital_type']} and accepting {h['insurance_providers']} insurance."
            for h in state.hospitals_found
        ])
        response_text = f"""I found the following hospitals for you for your following query:\n
            Query: {state.transcription.get('transcribed_text', 'N/A')}
            Hospital Types: {', '.join(state.recognition.get('hospital_type', [])) or 'Any'}
            Insurance: {', '.join(state.recognition.get('insurance', [])) or 'Any'}
            Location: {state.recognition.get('location') or 'Not specified'}
            Hospitals Found:\n
            {hospital_list_str}"""
        LOGGER.info(f"Generating response for {len(state.hospitals_found)} hospitals.")

    tts_result = await text_to_speech_tool.ainvoke({
        "text": response_text,
        "uid": state.uid
    })
    state.final_response = tts_result
    state.final_response_audio_path = tts_result["audio_path"]
    return state

graph.add_node("generate_response", generate_response)

# ----------------------------
# Conditional Edges
# ----------------------------
def check_query_completeness(state: HospitalFinderState):
    # Check if location coordinates are available
    if state.recognition and state.recognition.get("location_coordinates") and state.recognition["location_coordinates"][0] is not None:
        return "location_found"
    
    # If location is still missing after a few turns, maybe give up or try a generic search
    if state.turn_count >= MAX_TURNS: # Limit follow-up questions to 2 turns
        LOGGER.warning("Location still missing after multiple turns. Proceeding without location or ending.")
        return "max_turns_reached"
        
    return "location_missing"

graph.add_edge("transcriber", "recognizer")
graph.add_conditional_edges(
    "recognizer",
    check_query_completeness,
    {
        "location_missing": "ask_for_location",
        "location_found": "find_hospitals",
        "max_turns_reached": "generate_response" # Or a dedicated error node
    }
)

graph.add_edge("ask_for_location", "re_transcribe_user_response")
graph.add_edge("re_transcribe_user_response", "re_recognize_query")
graph.add_conditional_edges(
    "re_recognize_query",
    check_query_completeness,
    {
        "location_missing": "ask_for_location", # Ask again if still missing
        "location_found": "find_hospitals",
        "max_turns_reached": "generate_response"
    }
)

graph.add_edge("find_hospitals", "generate_response")
graph.add_edge("generate_response", END)

# ----------------------------
# Entry point
# ----------------------------
graph.set_entry_point("transcriber")

hospital_finder_graph = graph.compile()
