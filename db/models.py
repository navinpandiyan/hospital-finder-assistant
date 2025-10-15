from pydantic import BaseModel, Field
from typing import List, Optional

# -----------------------------
# Pydantic Models for LLM Output
# -----------------------------
class LLMResponseModel(BaseModel):
    intent: str = Field(default="find_hospital")
    location: Optional[str] = None
    hospital_type: List[str] = Field(default_factory=list)
    insurance: List[str] = Field(default_factory=list)
    
class HospitalFinderState(BaseModel):
    uid: str
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


class TTSResponseModel(BaseModel):
    dialogue: str
    tone: Optional[str] = None  # e.g., 'friendly', 'informative', 'empathetic'
