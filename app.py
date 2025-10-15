# app.py
import asyncio
import uuid  # Import uuid for generating unique IDs
from db.models import HospitalFinderState
from graphs.hospital_graph import hospital_finder_graph
from settings.config import LOGGER

async def main():
    state = HospitalFinderState()  # You can generate unique UID per user/session
    LOGGER.info("Starting Hospital Finder session...")

    # Run the compiled StateGraph
    final_state = await hospital_finder_graph.ainvoke(state)
    
    # Access final response
    # final_response = final_state.get("final_response", {})
    # if final_response:
    #     LOGGER.info(f"Final Response Text: {final_response.get("dialogue", final_response.get("text", ""))}")
    #     LOGGER.info(f"Final Response Audio Path: {final_state.get("final_response_audio_path", None)}")
    # else:
    #     LOGGER.warning("No final response generated.")

if __name__ == "__main__":
    asyncio.run(main())
