# rag_retrieve.py
# ---------------------
# Hospital retrieval using FAISS + RAG with optional fine-tuned LLM grounding

import os
import warnings
import math
from typing import List, Optional, Tuple, Dict

from db.models import RAGGroundedResponseModel
from settings.config import LOGGER, RAG_GROUNDER_MODEL, RAG_GROUNDER_TEMPERATURE, GROUND_WITH_FINE_TUNE, FINE_TUNE_OUTPUT_DIR as FINE_TUNE_MODEL_PATH
from settings.prompts import RAG_GROUNDER_SYSTEM_PROMPT, RAG_GROUNDER_USER_PROMPT
from settings.client import async_llm_client

import faiss
faiss.omp_set_num_threads(4)

# ---------------------------
# Suppress FAISS info messages
# ---------------------------
os.environ["FAISS_VERBOSE"] = "0"  # Suppress AVX2/AVX info prints
if os.name == "nt":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    warnings.filterwarnings("ignore", message=".*KMP_DUPLICATE_LIB_OK.*")

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from db.db import vector_db_path

# -----------------------------
# PEFT QLoRA imports
# -----------------------------
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

class HospitalRAGRetriever:
    def __init__(self, vector_db_path_override: str = None):
        self.vector_db_path = vector_db_path_override or vector_db_path
        self.embedding_model = OpenAIEmbeddings()
        self.vector_db = None
        self._load_vector_db()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if GROUND_WITH_FINE_TUNE:
            self._load_finetuned_model()

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
    # Optional fine-tuned LLM loader
    # -----------------------------
    def _load_finetuned_model(self):
        LOGGER.info(f"📦 Loading fine-tuned LLM from {FINE_TUNE_MODEL_PATH}")
        self.tokenizer = AutoTokenizer.from_pretrained(FINE_TUNE_MODEL_PATH, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            FINE_TUNE_MODEL_PATH,
            device_map="auto",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )
        # Wrap with PEFT if necessary
        try:
            self.model = PeftModel.from_pretrained(self.model, FINE_TUNE_MODEL_PATH)
        except Exception:
            LOGGER.info("No PEFT layers detected, using standard model.")
        self.model.to(self.device)
        self.model.eval()

    # -----------------------------
    # Utility functions
    # -----------------------------
    @staticmethod
    def _haversine_distance(lat1, lon1, lat2, lon2):
        R = 6371  # km
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat / 2) ** 2 +
             math.cos(math.radians(lat1)) *
             math.cos(math.radians(lat2)) *
             math.sin(dlon / 2) ** 2)
        return R * 2 * math.asin(math.sqrt(a))

    @staticmethod
    def _build_query(user_input: Dict) -> str:
        parts = []
        if user_input.get("user_loc"):
            parts.append(f"Hospitals located in {user_input['user_loc']}.")
        if user_input.get("hospital_names"):
            parts.append(f"Hospitals: {', '.join(user_input['hospital_names'])}.")
        if user_input.get("hospital_types"):
            parts.append(f"Specialties: {', '.join(user_input['hospital_types'])}.")
        if user_input.get("insurance_providers"):
            parts.append(f"Insurance accepted: {', '.join(user_input['insurance_providers'])}.")
        return " ".join(parts) if parts else "Hospitals nearby."

    # -----------------------------
    # Retrieval
    # -----------------------------
    def retrieve(self, user_input: Dict, extra_results: int = 5) -> List[Dict]:
        if not self.vector_db:
            raise RuntimeError("❌ Vector DB not loaded.")
        n_hospitals = user_input.get("n_hospitals", 5)
        user_lat = user_input["user_lat"]
        user_lon = user_input["user_lon"]
        max_distance = user_input.get("distance_km_radius", 300)
        intent = user_input.get("intent", "find_nearest")
        
        if intent in ["find_nearest", "find_best"]:
            query_text = self._build_query(user_input)
        else:
            query_text = user_input.get("user_query")
        
        top_docs = self.vector_db.similarity_search(query_text, k=int(n_hospitals + extra_results))

        if intent not in ["find_nearest", "find_best"]:
            filtered = [doc.metadata for doc in top_docs]                    
            return filtered
        else:
            filtered = []
            for doc in top_docs:
                meta = doc.metadata
                dist = self._haversine_distance(user_lat, user_lon, meta["latitude"], meta["longitude"])
                if dist <= max_distance:
                    filtered.append({**meta, "distance_km": round(dist, 2)})
            if intent == "find_best":
                return sorted(filtered, key=lambda x: (-x["rating"], x["distance_km"]))
            else:
                return sorted(filtered, key=lambda x: (x["distance_km"], -x["rating"]))

    # -----------------------------
    # Fine-tuned QLoRA grounding
    # -----------------------------
    async def ground_with_insurance_info_qlora(self, user_query: str, hospitals_context: List[dict]) -> str:
        if not hasattr(self, "model"):
            raise RuntimeError("Fine-tuned model not loaded. Set GROUND_WITH_FINE_TUNE=True")
        context_text = "\n".join([
            f"{h['hospital_name']} ({h['location']}), Specialties: {', '.join(h['hospital_type'])}, "
            f"Insurance: {', '.join(h['insurance_providers'])}, Rating: {h['rating']}"
            for h in hospitals_context
        ])
        prompt = f"Instruction: Answer the user query using the following hospital data.\n\nUser Query: {user_query}\n\nHospital Data:\n{context_text}\n\nAnswer:"

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=256)
        response_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return response_text
    
    # -----------------------------
    # Standard LLM grounding
    # -----------------------------
    async def ground_results(self, user_input: dict, retrieved_hospitals: List[dict]) -> RAGGroundedResponseModel:
        if not retrieved_hospitals:
            return RAGGroundedResponseModel(hospital_ids=[], dialogue="No hospitals found matching your criteria.")
        if user_input.get("intent", "find_nearest") in ["find_nearest", "find_best"]:
        # if user_input.get("intent"):
            hospital_context = "\n".join([
                f"{h['hospital_id']}: {h['hospital_name']} located in {h['location']}, "
                f"Specialties: {', '.join(h['hospital_type'])}, "
                f"Insurance accepted: {', '.join(h['insurance_providers'])}, "
                f"Rating: {h['rating']}, Distance: {h.get('distance_km', 'N/A')} km"
                for h in retrieved_hospitals
            ])

            user_message = RAG_GROUNDER_USER_PROMPT.format(
                user_query=user_input.get("user_query", ""),
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

            return response.choices[0].message.parsed
        
        else:
            results = await self.ground_with_insurance_info_qlora(user_input.get("user_query", ""), retrieved_hospitals)
            return results

    


# -----------------------------
# RAG wrapper
# -----------------------------
async def rag_search_wrapper(
    user_query: str,
    user_loc: str,
    user_lat: float,
    user_lon: float,
    intent: str = "find_nearest",
    hospital_types: Optional[List[str]] = None,
    hospital_names: Optional[str] = None,
    insurance_providers: Optional[List[str]] = None,
    n_hospitals: int = 5,
    distance_km_radius: float = 300,
    extra_results: int = 5,
) -> Tuple[List[dict], str]:
    retriever = HospitalRAGRetriever()

    user_input = {
        "user_query": user_query,
        "user_loc": user_loc,
        "user_lat": user_lat,
        "user_lon": user_lon,
        "intent": intent,
        "hospital_names": hospital_names,
        "hospital_types": hospital_types or [],
        "insurance_providers": insurance_providers or [],
        "n_hospitals": n_hospitals,
        "distance_km_radius": distance_km_radius,
        "extra_results": extra_results,
    }

    retrieved = retriever.retrieve(user_input, extra_results=extra_results)
    grounded = await retriever.ground_results(user_input, retrieved)
    id_to_hospital = {h["hospital_id"]: h for h in retrieved}
    selected_hospitals = [id_to_hospital[h_id] for h_id in grounded.hospital_ids if h_id in id_to_hospital]

    if GROUND_WITH_FINE_TUNE and selected_hospitals:
        dialogue = await retriever.ground_with_insurance_info_qlora(user_query, selected_hospitals)
    else:
        dialogue = grounded.dialogue

    return selected_hospitals, dialogue


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    import asyncio
    GROUND_WITH_FINE_TUNE = False  # toggle fine-tuned LLM

    test_input = {
        'user_query': 'can i use medlife insurance at fujairah diagnostics center?', 
        'user_loc': 'fujairah', 
        'user_lat': 25.1244604, 
        'user_lon': 56.3355085, 
        'intent': 'get_insurance_coverage', 
        'hospital_names': ['medlife'], 
        'hospital_types': [], 
        'insurance_providers': 1, 
        'n_hospitals': 300.0, 
        'distance_km_radius': 5,
        'extra_results': 5
        }

    hospitals, dialogue = asyncio.run(rag_search_wrapper(**test_input))

    print("\nSelected Hospitals:")
    for h in hospitals:
        print(h)

    print("\nDialogue for TTS:")
    print(dialogue)
