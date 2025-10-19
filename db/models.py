import uuid
from pydantic import BaseModel, Field
from typing import List, Optional

# -----------------------------
# Pydantic Models for LLM Output
# -----------------------------
class LLMResponseModel(BaseModel):
    query: str
    intent: str = Field(
        default="find_nearest",
        description="Intent type: find_nearest | find_best | find_by_insurance | find_by_hospital"
    )
    location: Optional[str] = None
    hospital_names: Optional[List[str]] = None
    hospital_type: List[str] = Field(default_factory=list)
    insurance: List[str] = Field(default_factory=list)
    provider_name: Optional[str] = None
    n_hospitals: Optional[int] = 5
    distance_km: Optional[float] = 30000

    
# -----------------------------
# Pydantic model for structured RAG grounding response
# -----------------------------
class RAGGroundedResponseModel(BaseModel):
    hospital_ids: List[int]
    dialogue: str
    
class HospitalFinderState(BaseModel):
    uid: str = Field(default_factory=lambda: str(uuid.uuid4()))
    input_audio_path: Optional[str] = None
    transcription: Optional[dict] = None
    recognition: Optional[dict] = None

    clarify_user_response_audio_path: Optional[str] = None
    clarify_transcription: Optional[dict] = None
    clarify_recognition: Optional[dict] = None

    hospitals_found: Optional[dict] = {}
    clarify_bot_response_audio_path: Optional[str] = None
    turn_count: int = 0
    last_question: Optional[str] = None
    
    
    final_response: Optional[dict] = None
    final_response_text: Optional[str] = None
    final_response_audio_path: Optional[str] = None
    
    user_wants_exit: bool = False


class TTSResponseModel(BaseModel):
    dialogue: str
    tone: Optional[str] = None  # e.g., 'friendly', 'informative', 'empathetic'
