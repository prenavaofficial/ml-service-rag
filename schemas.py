from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class RAGQueryRequest(BaseModel):
    query: Optional[str] = Field(None, description="Pertanyaan teks (standar Prenava)")
    text: Optional[str] = Field(None, description="Pertanyaan teks (alias backward compatible)")

    def get_effective_query(self) -> str:
        return (self.query or self.text or "").strip()


class RetrievedContextItem(BaseModel):
    page: int
    source: str
    similarity: float
    danger_level: str
    text_snippet: str


class RAGResponse(BaseModel):
    status: str = Field(..., description="'success' atau 'emergency_alert'")
    query: str
    transcribed_text: Optional[str] = None
    answer: str
    reference_source: str
    is_emergency: bool
    danger_level: str
    provider_used: str  # "groq" atau "gemini" atau "circuit_breaker"
    recommended_action: str
    danger_keywords: List[str] = []
    retrieved_contexts: List[RetrievedContextItem] = []
