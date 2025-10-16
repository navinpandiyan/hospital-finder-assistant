import uuid
from pydantic import BaseModel, Field
from typing import List, Optional

# -----------------------------
# Pydantic Models for LLM Output
# -----------------------------
class LLMResponseModel(BaseModel):
    intent: str = Field(default="find_nearest")
    location: Optional[str] = None
    hospital_type: List[str] = Field(default_factory=list)
    insurance: List[str] = Field(default_factory=list)
    n_hospitals: Optional[int] = 5 
    distance_km: Optional[float] = 300
    
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

    hospitals_found: Optional[List[dict]] = None
    clarify_bot_response_audio_path: Optional[str] = None
    turn_count: int = 0
    last_question: Optional[str] = None
    final_response: Optional[dict] = None
    final_response_audio_path: Optional[str] = None
    
    user_wants_exit: bool = False


class TTSResponseModel(BaseModel):
    dialogue: str
    tone: Optional[str] = None  # e.g., 'friendly', 'informative', 'empathetic'
