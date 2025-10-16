# rag_retrieve.py
# ---------------------
# Hospital retrieval using FAISS + RAG with async LLM grounding

import os
import warnings
from typing import List, Optional, Tuple
import math
from typing import List, Dict

from db.models import RAGGroundedResponseModel
from settings.config import LOGGER, RAG_GROUNDER_MODEL, RAG_GROUNDER_TEMPERATURE
from settings.prompts import RAG_GROUNDER_SYSTEM_PROMPT, RAG_GROUNDER_USER_PROMPT
from settings.client import async_llm_client

# 🧩 Fix FAISS OpenMP conflict on Windows
if os.name == "nt":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    warnings.filterwarnings("ignore", message=".*KMP_DUPLICATE_LIB_OK.*")

import faiss
faiss.omp_set_num_threads(4)

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from db.db import vector_db_path


class HospitalRAGRetriever:
    def __init__(self, vector_db_path_override: str = None):
        self.vector_db_path = vector_db_path_override or vector_db_path
        self.embedding_model = OpenAIEmbeddings()
        self.vector_db = None
        self._load_vector_db()

    # -----------------------------
    # Load FAISS DB
    # -----------------------------
    def _load_vector_db(self):
        if not os.path.exists(self.vector_db_path):
            raise FileNotFoundError(f"❌ FAISS vector DB not found at {self.vector_db_path}")

        self.vector_db = FAISS.load_local(
            self.vector_db_path,
            self.embedding_model,
            allow_dangerous_deserialization=True
        )
        LOGGER.info(f"✅ Loaded FAISS vector DB from {self.vector_db_path}")

    # -----------------------------
    # Haversine distance calculator
    # -----------------------------
    @staticmethod
    def _haversine_distance(lat1, lon1, lat2, lon2):
        R = 6371  # km
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(math.radians(lat1))
            * math.cos(math.radians(lat2))
            * math.sin(dlon / 2) ** 2
        )
        return R * 2 * math.asin(math.sqrt(a))

    # -----------------------------
    # Build embedding query (semantic only)
    # -----------------------------
    @staticmethod
    def _build_query(user_input: Dict) -> str:
        """
        Build a semantically aligned query for FAISS retrieval,
        matching the phrasing used during vector embedding.
        """
        parts = []

        if user_input.get("user_loc"):
            parts.append(f"Hospitals located in {user_input['user_loc']}.")
        if user_input.get("hospital_types"):
            parts.append(f"Specialties: {', '.join(user_input['hospital_types'])}.")
        if user_input.get("insurance_providers"):
            parts.append(f"Insurance accepted: {', '.join(user_input['insurance_providers'])}.")

        return " ".join(parts) if parts else "Hospitals nearby."

    # -----------------------------
    # Retrieve hospitals
    # -----------------------------
    def retrieve(self, user_input: Dict, extra_results: int = 5) -> List[Dict]:
        """
        Retrieve top hospitals based on semantic similarity + distance filtering.
        Intent ('find_nearest' or 'find_best') affects sorting only.
        """
        if not self.vector_db:
            raise RuntimeError("❌ Vector DB not loaded.")

        n_hospitals = user_input.get("n_hospitals", 5)
        user_lat = user_input["user_lat"]
        user_lon = user_input["user_lon"]
        max_distance = user_input.get("distance_km_radius", 300)
        intent = user_input.get("intent", "find_nearest")

        query_text = self._build_query(user_input)
        top_docs = self.vector_db.similarity_search(query_text, k=n_hospitals + extra_results)

        filtered = []
        for doc in top_docs:
            meta = doc.metadata
            dist = self._haversine_distance(user_lat, user_lon, meta["latitude"], meta["longitude"])
            if dist <= max_distance:
                filtered.append({**meta, "distance_km": round(dist, 2)})

        # Intent-based sorting
        if intent == "find_best":
            filtered_sorted = sorted(filtered, key=lambda x: (-x["rating"], x["distance_km"]))
        else:
            filtered_sorted = sorted(filtered, key=lambda x: (x["distance_km"], -x["rating"]))

        return filtered_sorted

    # -----------------------------
    # Async function to ground RAG results via LLM
    # -----------------------------
    async def ground_results(self, user_input: dict, retrieved_hospitals: List[dict]) -> RAGGroundedResponseModel:
        """
        Take RAG-retrieved hospital data + user input, send to LLM,
        and return a structured response with relevant hospital_ids and TTS dialogue.
        """
        if not retrieved_hospitals:
            return RAGGroundedResponseModel(
                hospital_ids=[],
                dialogue="No hospitals found matching your criteria."
            )

        hospital_context = "\n".join([
            f"{h['hospital_id']}: {h['hospital_name']} located in {h['city']}, "
            f"Specialties: {', '.join(h['hospital_type'])}, "
            f"Insurance accepted: {', '.join(h['insurance_providers'])}, "
            f"Rating: {h['rating']}, Distance: {h.get('distance_km', 'N/A')} km"
            for h in retrieved_hospitals
        ])

        user_message = RAG_GROUNDER_USER_PROMPT.format(
            user_loc=user_input.get("user_loc", ""),
            user_lat=user_input.get("user_lat", ""),
            user_lon=user_input.get("user_lon", ""),
            intent=user_input.get("intent", "find_nearest"),
            specialties=", ".join(user_input.get("hospital_types", [])),
            insurance_providers=", ".join(user_input.get("insurance_providers", [])),
            n_hospitals=user_input.get("n_hospitals", 5),
            distance_km_radius=user_input.get("distance_km_radius", 300),
            hospital_context=hospital_context
        )

        response = await async_llm_client.beta.chat.completions.parse(
            model=RAG_GROUNDER_MODEL,
            messages=[
                {"role": "system", "content": RAG_GROUNDER_SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            temperature=RAG_GROUNDER_TEMPERATURE,
            response_format=RAGGroundedResponseModel
        )

        llm_output: RAGGroundedResponseModel = response.choices[0].message.parsed
        return llm_output


# -----------------------------
# RAG wrapper for LangChain tools
# -----------------------------
async def rag_search_wrapper(
    user_loc: str,
    user_lat: float,
    user_lon: float,
    intent: str = "find_nearest",  # find_nearest / find_best
    hospital_types: Optional[List[str]] = None,
    insurance_providers: Optional[List[str]] = None,
    n_hospitals: int = 5,
    distance_km_radius: float = 300,
    extra_results: int = 5,
) -> Tuple[List[dict], str]:
    """
    Performs a full RAG-based hospital lookup:
    1. Retrieves hospitals via FAISS vector DB based on semantic similarity.
    2. Filters and sorts by distance/rating depending on user intent.
    3. Sends retrieved hospitals + user info to LLM for grounding.
    Returns a Pydantic model with hospital_ids and dialogue suitable for TTS.
    """
    retriever = HospitalRAGRetriever()

    user_input = {
        "user_loc": user_loc,
        "user_lat": user_lat,
        "user_lon": user_lon,
        "intent": intent,
        "hospital_types": hospital_types or [],
        "insurance_providers": insurance_providers or [],
        "n_hospitals": n_hospitals,
        "distance_km_radius": distance_km_radius,
        "extra_results": extra_results,
    }

    # 1️⃣ Retrieve hospitals from FAISS
    retrieved = retriever.retrieve(user_input, extra_results=extra_results)

    # 2️⃣ Ground results using LLM
    grounded: RAGGroundedResponseModel = await retriever.ground_results(user_input, retrieved)

    # 3️⃣ Map hospital_ids back to full hospital dicts
    id_to_hospital = {h["hospital_id"]: h for h in retrieved}
    selected_hospitals = [id_to_hospital[h_id] for h_id in grounded.hospital_ids if h_id in id_to_hospital]

    # 4️⃣ Return tuple: (list of hospital dicts, dialogue)
    return selected_hospitals, grounded.dialogue

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    import asyncio

    test_input = {
        "user_loc": "dubai",
        "user_lat": 25.2048,
        "user_lon": 55.2708,
        "intent": "find_best",
        "hospital_types": ["urology", "oncology"],
        "insurance_providers": ["adnic", "daman"],
        "n_hospitals": 3,
        "distance_km_radius": 300,
        "extra_results": 5
    }
    
    final_hospitals, dialogue = asyncio.run(rag_search_wrapper(**test_input))

    print("Selected Hospitals:")
    for h in final_hospitals:
        print(h)

    print("\nDialogue for TTS:")
    print(dialogue)

