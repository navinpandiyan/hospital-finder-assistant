"""
tools/recognizer.py
-------------------
Query Understanding Module for AI Voice Bot – Nearest Network Hospital Finder

Objective:
-----------
Extract key entities from transcribed user queries:
- Location
- Hospital Type / Specialty
- Insurance Provider (if mentioned)

Assumption:
------------
Intent is always 'find_hospital'
"""

from typing import Optional, Dict
import uuid
import asyncio
from settings.config import (
    DEFAULT_N_HOSPITALS_TO_RETURN,
    DEFAULT_DISTANCE_KM,
    LOGGER,
    NLP_MODEL,
    HOSPITAL_TYPES,
    INSURANCE_PROVIDERS,
    FUZZY_MATCH_THRESHOLD,
    RECOGNIZER_MODEL,
    RECOGNIZER_TEMPERATURE
)
from db.models import LLMResponseModel
from rapidfuzz import process, fuzz
from settings.client import async_llm_client
from settings.prompts import (
    RECOGNIZER_SYSTEM_PROMPT, 
    RECOGNIZER_USER_PROMPT
)
from utils.utils import get_lat_long


class QueryRecognizer:
    """
    Extracts key entities (location, hospital_type, insurance) from a transcribed query.
    """

    def __init__(self, hospital_types: Optional[list] = HOSPITAL_TYPES, insurance_providers: Optional[list] = INSURANCE_PROVIDERS):
        if NLP_MODEL is None:
            LOGGER.error("QueryRecognizer not loaded.")
            raise ValueError("NLP_MODEL not loaded. Please check config.py")
        self.nlp = NLP_MODEL
        

        # Common hospital specialties (expandable)
        self.hospital_types = hospital_types

        # Common insurance providers (expandable)
        self.insurance_providers = insurance_providers

    # -----------------------------
    # Extraction Methods
    # -----------------------------

    def _extract_location(self, doc) -> Optional[str]:
        """Extract geographic or facility-based location from query."""
        for ent in doc.ents:
            if ent.label_ in ["GPE", "LOC", "FAC"]:
                return ent.text
        return None

    def _extract_hospital_types(self, text: str) -> Optional[list]:
        """
        Detects all hospital specialties using fuzzy matching.
        Can match abbreviations or partial words (e.g., "cardio" → "cardiology").
        Returns a list of matched hospital types.
        """
        text_lower = text.lower()
        matches = process.extract(
            query=text_lower,
            choices=self.hospital_types,
            scorer=fuzz.partial_ratio,
            limit=1  # return all matches
        )

        # Keep matches with similarity above threshold (70)
        hospital_matches = [match[0] for match in matches if match[1] >= FUZZY_MATCH_THRESHOLD]

        # Remove duplicates while preserving order
        seen = set()
        hospital_matches = [x for x in hospital_matches if not (x in seen or seen.add(x))]

        if hospital_matches:
            return hospital_matches
        return []

    def _extract_insurance_providers(self, text: str) -> Optional[list]:
        """
        Detects all mentioned insurance providers using fuzzy matching.
        Can match partial names or abbreviations (e.g., "ADNIC" → "adnic").
        Returns a list of matched providers.
        """
        text_lower = text.lower()
        matches = process.extract(
            query=text_lower,
            choices=self.insurance_providers,
            scorer=fuzz.partial_ratio,
            limit=None  # return all matches
        )

        # Keep matches with similarity above threshold
        insurance_matches = [match[0] for match in matches if match[1] >= FUZZY_MATCH_THRESHOLD]

        # If user mentions 'insurance' generically but no provider matched, keep 'mentioned'
        if "insurance" in text_lower and not insurance_matches:
            insurance_matches.append("mentioned")

        # Remove duplicates while preserving order
        seen = set()
        insurance_matches = [x for x in insurance_matches if not (x in seen or seen.add(x))]

        if insurance_matches:
            return insurance_matches
        return []

    # -----------------------------
    # LLM-based Extraction
    # -----------------------------
    async def _extract_with_llm(self, query_text: str) -> LLMResponseModel:
        """
        Use an LLM to extract structured entities from the query.
        Returns a validated Pydantic model.
        """

        user_message = RECOGNIZER_USER_PROMPT.format(query_text=query_text)
        
        response = await async_llm_client.beta.chat.completions.parse(
            model=RECOGNIZER_MODEL,
            messages=[
                {"role": "system", "content": RECOGNIZER_SYSTEM_PROMPT},
                {"role": "user", "content": user_message}],
            temperature=RECOGNIZER_TEMPERATURE,
            response_format=LLMResponseModel
        )

        llm_output = response.choices[0].message.parsed

        # The parsed output is already available as a Pydantic model
        llm_output: LLMResponseModel = response.choices[0].message.parsed

        # Ensure all string values are lowercase
        llm_output.location = llm_output.location.lower() if llm_output.location else None
        llm_output.hospital_type = [h.lower() for h in llm_output.hospital_type]
        llm_output.insurance = [i.lower() for i in llm_output.insurance]
        llm_output.n_hospitals = llm_output.n_hospitals or DEFAULT_N_HOSPITALS_TO_RETURN
        llm_output.distance_km = llm_output.distance_km or DEFAULT_DISTANCE_KM

        # Return the Pydantic model (or dict if needed)
        return llm_output

    # -----------------------------
    # Public Recognize Method
    # -----------------------------
    async def recognize(self, query_text: str, uid: str, use_llm: bool = False) -> Dict:
        """
        Main method to extract entities from query.
        use_llm: If True, use LLM; otherwise, use spaCy + fuzzy matching.
        """
        if not query_text:
            query_text = "No Response Recieved. Exiting."

        if use_llm:
            llm_result = (await self._extract_with_llm(query_text)).model_dump()
            
            # Ensure consistency with non-LLM mode
            result = {
                "uid": uid or str(uuid.uuid4()),
                "query": query_text,
                "output_query": llm_result.get("query", query_text),
                "intent": llm_result.get("intent", "find_nearest"),
                "location": llm_result.get("location"),
                "location_coordinates": await asyncio.to_thread(get_lat_long, llm_result.get("location")) if llm_result.get("location") else (None, None),
                "hospital_names": llm_result.get("hospital_names", []),
                "hospital_type": llm_result.get("hospital_type", []),
                "insurance": llm_result.get("insurance", []),
                "n_hospitals": llm_result.get("n_hospitals", DEFAULT_N_HOSPITALS_TO_RETURN),
                "distance_km": llm_result.get("distance_km", DEFAULT_DISTANCE_KM),
            }
            
        else:
            # Default spaCy + fuzzy
            doc = await asyncio.to_thread(self.nlp, query_text)
            location = self._extract_location(doc)
            hospital_type = self._extract_hospital_types(query_text)
            insurance = self._extract_insurance_providers(query_text)

            result = {
                "uid": uid or str(uuid.uuid4()),
                "query": query_text,
                "intent": "find_nearest",  # Default for non-LLM mode
                "location": location,
                "location_coordinates": await asyncio.to_thread(get_lat_long, location) if location else (None, None),
                "hospital_type": hospital_type,
                "insurance": insurance,
                "n_hospitals": DEFAULT_N_HOSPITALS_TO_RETURN,  # Default for non-LLM mode
                "distance_km": DEFAULT_DISTANCE_KM,  # Default for non-LLM mode
            }

        return result

async def recognize_wrapper(
    query_text: str,
    uid: Optional[str] = None,
    use_llm: bool = False,
) -> Dict:
    """
    Convenience wrapper to recognize entities from query.
    Parameters
    ----------
    query_text : str
        The transcribed user query
    uid : str, optional
        Predefined unique ID (else autogenerated)
    use_llm : bool, default False
        Whether to use LLM for extraction (True) or spaCy + fuzzy (False)

    Returns
    -------
    dict : {
        "uid": str,
        "query": str,
        "intent": str,
        "location": str or None,
        "location_coordinates": (lat, long) or (None, None),
        "hospital_type": list of str,
        "insurance": list of str,
        "n_hospitals": int,
        "distance_km": int or float
    }
    """
    recognizer = QueryRecognizer()
    return await recognizer.recognize(query_text, uid=uid, use_llm=use_llm)

# if __name__ == "__main__":
#     async def test_recognize():
#         # Quick test run
#         recognizer = QueryRecognizer()

#         test_queries = [
#             "Find me cardio hospitals in Dubai",
#             "Show me the nearest orthopaedic and dental hospitals near Alqusaidat",
#             "Are there any ENT hospitals around me?",
#             "Find the nearest cardiology hospital in Abu Dhabi with Aetna insurance",
#             "Find the best dental hospital covered by ADNIC",
#         ]

#         res = []
#         for q in test_queries:
#             print("\nQuery:", q)
#             result = await recognizer.recognize(q, uid=str(uuid.uuid4()), use_llm=True)
#             res.append(result)
#             print("Extracted:", result)
            
#             with open("test_recognizer_output.json", "w") as f:
#                 import json
#                 json.dump(res, f, indent=4)
#     asyncio.run(test_recognize())

#     test_queries = [
#         "Find me cardio hospitals in Dubai",
#         "Show me the nearest orthopaedic and dental hospitals near Alqusaidat",
#         "Are there any ENT hospitals around me?",
#         "Find the nearest cardiology hospital in Abu Dhabi with Aetna insurance",
#         "Find the best dental hospital covered by ADNIC",
#     ]

#     res = []
#     for q in test_queries:
#         print("\nQuery:", q)
#         result = recognizer.recognize(q, uid=str(uuid.uuid4()), use_llm=True)
#         res.append(result)
#         print("Extracted:", result)
        
#         with open("test_recognizer_output.json", "w") as f:
#             import json
#             json.dump(res, f, indent=4)
