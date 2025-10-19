# rag_retrieve.py
# ---------------------
# Hospital retrieval using FAISS + RAG with optional fine-tuned LLM grounding

import os
import warnings
import math
from typing import List, Optional, Tuple, Dict

from db.models import RAGGroundedResponseModel
from settings.config import INSURANCE_PROVIDERS, LOGGER, RAG_GROUNDER_MODEL, RAG_GROUNDER_TEMPERATURE, GROUND_WITH_FINE_TUNE, FINE_TUNE_OUTPUT_DIR as FINE_TUNE_MODEL_PATH
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

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(FINE_TUNE_MODEL_PATH, use_fast=True)

        # Load PEFT-wrapped model directly
        try:
            # Directly load PEFT model (safest, suppresses duplicate adapter warnings)
            self.model = PeftModel.from_pretrained(
                FINE_TUNE_MODEL_PATH,
                device_map="cuda:0",
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
            )
        except Exception:
            # Fallback: if no PEFT layers exist, load standard model
            LOGGER.info("No PEFT layers detected, loading base model.")
            self.model = AutoModelForCausalLM.from_pretrained(
                FINE_TUNE_MODEL_PATH,
                device_map="cuda:0",
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
            )

        self.model = torch.compile(self.model)
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
    def retrieve(self, user_input: Dict, extra_results: int = 2) -> List[Dict]:
        if not self.vector_db:
            raise RuntimeError("❌ Vector DB not loaded.")
        n_hospitals = user_input.get("n_hospitals", 5)
        user_loc = user_input.get("user_loc", "")
        user_lat = user_input["user_lat"]
        user_lon = user_input["user_lon"]
        max_distance = user_input.get("distance_km_radius", 30000)
        intent = user_input.get("intent", "find_nearest")
        user_hospitals = user_input.get("hospital_names", [])
        user_specialities = user_input.get("hospital_types", [])
        user_insurances = user_input.get("insurance_providers", [])
        user_insurances = list(set(user_insurances).intersection(INSURANCE_PROVIDERS))

        if user_loc or user_hospitals or user_specialities or user_insurances:
            query_text = self._build_query(user_input)
            
            if intent in ["find_by_hospital", "compare_hospitals"]:
                k = n_hospitals
            else:
                k = int(n_hospitals + extra_results)
        else:
            return []
        
        top_docs = self.vector_db.similarity_search(query_text, k=k)
        LOGGER.info(f"RAG Query: {query_text}")
        LOGGER.info(f"RAG Retrieved: {len(top_docs)}")

        if not user_loc:
            return [doc.metadata for doc in top_docs]

        filtered = []
        for doc in top_docs:
            meta = doc.metadata
            if user_lat and user_lon:
                dist = self._haversine_distance(user_lat, user_lon, meta["latitude"], meta["longitude"])
                if dist <= max_distance:
                    filtered.append({**meta, "distance_km": round(dist, 2)})

        LOGGER.info(f"RAG Filtered: {len(filtered)}")
        if intent == "find_best":
            key_func = lambda x: (-x["rating"], x["distance_km"])
        else:
            key_func = lambda x: (x["distance_km"], -x["rating"])
        return sorted(filtered, key=key_func)

    # -----------------------------
    # Fine-tuned QLoRA grounding
    # -----------------------------
    async def ground_with_insurance_info_qlora(self, user_query: str, retrieved_hospitals: List[dict]) -> str:
        if not hasattr(self, "model"):
            raise RuntimeError("Fine-tuned model not loaded. Set GROUND_WITH_FINE_TUNE=True")

        # Build context string
        hospital_context = "\n".join([
            f"{h['hospital_name']} located in {h['location']}, "
            f"Specialties: {', '.join(h['hospital_type'])}, "
            f"Rating: {h['rating']} "
            f"Insurance accepted: {', '.join(h['insurance_providers'])}"
            for h in retrieved_hospitals
        ])

        prompt = (
            f"### Instruction:\n{user_query.strip()}\n\n"
            f"### Context:\n{hospital_context}\n\n"
            f"### Response:\n"
        )

        # Tokenize only once on GPU
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding="longest"
        ).to(self.device)

        # Enable faster inference: disable gradients, enable caching, and use half precision
        with torch.inference_mode():  # slightly faster than no_grad()
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=256,         # keep response concise
                do_sample=True,
                temperature=0.5,
                top_p=0.9,
                use_cache=True,             # enables past_key_values cache
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode output efficiently
        response_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        if "." in response_text:
            response_text = response_text[:response_text.rfind(".") + 1].strip()
        response = response_text.split("### Response:\n")[-1].strip() 
        return response


    
    # -----------------------------
    # Standard LLM grounding
    # -----------------------------
    async def ground_results(self, user_input: dict, retrieved_hospitals: List[dict]) -> RAGGroundedResponseModel:
        user_loc = user_input.get("user_loc", "")
        user_hospitals = user_input.get("hospital_names", [])
        user_specialities = user_input.get("hospital_types", [])
        user_insurances = user_input.get("insurance_providers", [])
        user_insurances = list(set(user_insurances).intersection(INSURANCE_PROVIDERS))
        
        if user_input.get("intent", "find_nearest") not in ["find_by_hospital", "find_by_insurance"]:  
        # if user_input.get("intent", "find_nearest"):  
        # if not GROUND_WITH_FINE_TUNE:  
            LOGGER.info(f"RAG Grounding: {RAG_GROUNDER_MODEL}")
            if not retrieved_hospitals:
                return RAGGroundedResponseModel(hospital_ids=[], dialogue="No hospitals found matching your criteria.")
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
                distance_km_radius=user_input.get("distance_km_radius", 30000),
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
            
            response_parsed = response.choices[0].message.parsed
            return response_parsed
        
        else: #Use QLoRA Model if find_by_insurance / find_by_hospital
            LOGGER.info(f"RAG Grounding: QLoRA Model")
            response = await self.ground_with_insurance_info_qlora(user_input.get("user_query", ""), retrieved_hospitals)
            result = RAGGroundedResponseModel(hospital_ids=[h['hospital_id'] for h in retrieved_hospitals], dialogue=response)
            return result

    


# -----------------------------
# RAG wrapper
# -----------------------------
async def rag_search_wrapper(
    retriever: HospitalRAGRetriever, # Accept pre-initialized retriever
    user_query: Optional[str] = None,
    user_loc: Optional[str] = None,
    user_lat: Optional[float] = None,
    user_lon: Optional[float] = None,
    intent: Optional[str] = "find_nearest",
    hospital_types: Optional[List[str]] = None,
    hospital_names: Optional[str] = None,
    insurance_providers: Optional[List[str]] = None,
    n_hospitals: int = 5,
    distance_km_radius: float = 30000,
    extra_results: int = 5,
) -> Tuple[List[dict], str]:
    
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
    return retrieved, selected_hospitals, grounded.dialogue


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    import asyncio
    GROUND_WITH_FINE_TUNE = True  # toggle fine-tuned LLM
    
    retriever = HospitalRAGRetriever()

    test_input = {
        'user_query': "Can you list the insurance options for Al Awir Dental Hospital? Do they accept Direct Billing?", 
        'user_loc': "", 
        'user_lat': None, 
        'user_lon': None, 
        'intent': "find_by_hospital", 
        'hospital_names': ["Al Awir Dental Hospital"], 
        'hospital_types': [], 
        'insurance_providers': [], 
        'n_hospitals': 1, 
        'distance_km_radius': 500,
        'extra_results': 5
        }

    retrieved, selected, dialogue = asyncio.run(rag_search_wrapper(**test_input, retriever=retriever))

    print("\nSelected Hospitals:")
    for h in selected:
        print(h)

    print("\nDialogue for TTS:")
    print(dialogue)
