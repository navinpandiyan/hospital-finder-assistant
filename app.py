# app.py
import asyncio
import time
import sys
import uuid
from db.models import HospitalFinderState
from graphs.hospital_graph import hospital_finder_graph
from settings.config import LOGGER
from tools.rag_retrieve import HospitalRAGRetriever
from graphs.graph_tools import set_rag_retriever_instance
from utils.utils import show_initializing_animation





async def main():
    # --- Step 1: Initialization animation ---
    show_initializing_animation()

    # --- Step 2: Setup RAG retriever ---
    rag_retriever = HospitalRAGRetriever()
    set_rag_retriever_instance(rag_retriever)

    # --- Step 3: Prepare state ---
    state = HospitalFinderState()
    LOGGER.info("Starting Hospital Finder session...")

    # --- Step 4: Enter chat room animation ---
    print("\n💬 Entering chat room...")
    await asyncio.sleep(1)
    print("Bot: Hello! I am your hospital assistant. How can I help you today?")

    # --- Step 5: Run LangGraph ---
    final_state = await hospital_finder_graph.ainvoke(state)


if __name__ == "__main__":
    asyncio.run(main())
