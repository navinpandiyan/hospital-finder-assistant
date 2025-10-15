# app.py
import asyncio
from db.models import HospitalFinderState
from agents.hospital_graph import hospital_finder_graph
from settings.config import LOGGER

async def main():
    # Create a new state for the user session
    state = HospitalFinderState(uid="user_123")  # You can generate unique UID per user/session
    LOGGER.info("Starting Hospital Finder session...")

    # Run the compiled StateGraph
    final_state = await hospital_finder_graph.ainvoke(state)
    
    final_response = final_state.get("final_response", {})
    breakpoint()
    # Access final response
    if final_response:
        LOGGER.info(f"Final Response Text: {final_response.get("dialogue", final_response.get("text", ""))}")
        LOGGER.info(f"Final Response Audio Path: {getattr(final_state, 'final_response_audio_path', None)}")
    else:
        LOGGER.warning("No final response generated.")

if __name__ == "__main__":
    asyncio.run(main())
