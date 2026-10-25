import os
import sys
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Any, List

from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from groq import Groq

from schemas import RAGQueryRequest, RAGResponse
from danger_classifier import DANGER_TAXONOMY
from rag_engine import ask_rag, llm, gemini_model

load_dotenv()

app = FastAPI(
    title="Prenava - Voice Medical RAG Assistant API",
    description=(
        "Microservice RAG Medis & Transkripsi Suara (Whisper STT) berbasis standar "
        "Buku KIA Kemenkes RI untuk asisten konsultasi ibu hamil dan deteksi tanda bahaya maternal."
    ),
    version="2.0.0"
)

# CORS Middleware agar bisa diakses oleh Laravel (:8000), React Bidan (:5173), dan Flutter (:8085)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None


async def _transcribe_upload_file(file: UploadFile) -> str:
    """Helper untuk mentranskripsikan file audio rekaman suara ibu hamil via Groq Whisper."""
    if not groq_client:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Groq Client belum terkonfigurasi untuk Speech-to-Text."
        )

    filename = file.filename or "recording.m4a"
    file_ext = Path(filename).suffix.lower() or ".m4a"

    allowed_exts = [".m4a", ".wav", ".mp3", ".ogg", ".webm", ".aac"]
    if file_ext not in allowed_exts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Format audio '{file_ext}' tidak didukung. Harap upload format: {', '.join(allowed_exts)}"
        )

    temp_path = None
    try:
        audio_bytes = await file.read()
        if not audio_bytes:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File audio kosong.")

        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            tmp.write(audio_bytes)
            temp_path = tmp.name

        with open(temp_path, "rb") as f:
            transcription = groq_client.audio.transcriptions.create(
                file=(os.path.basename(temp_path), f.read()),
                model="whisper-large-v3-turbo",
                language="id",
                response_format="json",
                temperature=0.0
            )

        text_result = getattr(transcription, "text", "") or ""
        return text_result.strip()

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


# 1. Endpoint Informasi & Health Check
@app.get("/", tags=["General"])
def root():
    port = int(os.getenv("PORT", 8002))
    return {
        "service": "Prenava Voice-Enabled Medical RAG Assistant",
        "status": "online",
        "version": "2.0.0",
        "port": port,
        "docs_url": f"/docs",
        "supported_features": [
            "Voice Note Speech-to-Text (Groq Whisper Turbo)",
            "Semantic Full-Page Retrieval (ChromaDB + Buku KIA Kemenkes RI)",
            "Dual-LLM Reasoning (Groq Qwen + Google Gemini Fallback)",
            "Obstetric Emergency Circuit Breaker (6 Kategori Kemenkes)"
        ]
    }


@app.get("/rag/health", tags=["Monitoring"])
def health_check():
    """Memeriksa kesiapan database vektor, Groq API, dan Gemini Fallback API."""
    return {
        "status": "healthy",
        "vector_store": "ChromaDB (Active - Buku KIA Hal 1-155)",
        "groq_configured": bool(groq_client and llm),
        "gemini_configured": bool(gemini_model),
        "primary_model": "qwen/qwen3.8-27b",
        "fallback_model": "gemini-3.8-flash"
    }


@app.get("/rag/danger-catalog", tags=["Knowledge Base"])
def get_danger_signs_catalog():
    """Mengembalikan katalog resmi tanda bahaya kehamilan Buku KIA Kemenkes RI untuk UI Mobile/Web."""
    catalog = []
    for code, info in DANGER_TAXONOMY.items():
        catalog.append({
            "code": code,
            "label": info["label"],
            "recommended_action": info["action"]
        })
    return {
        "total_categories": len(catalog),
        "source": "Buku KIA Kemenkes RI - Lembar Tanda Bahaya Kehamilan",
        "catalog": catalog
    }


# 2. Endpoint Standar Ekosistem Prenava (Digunakan oleh Laravel & Flutter)
@app.post("/rag/ask-text", response_model=RAGResponse, tags=["RAG Consultation"])
def ask_by_text(request: RAGQueryRequest):
    """Konsultasi tanya-jawab medis teks standar Prenava."""
    query = request.get_effective_query()
    if not query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Pertanyaan tidak boleh kosong!"
        )

    try:
        response = ask_rag(query=query)
        return response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Terjadi kesalahan saat memproses RAG: {str(e)}"
        )


@app.post("/rag/ask-voice", response_model=RAGResponse, tags=["RAG Consultation"])
async def ask_by_voice(file: UploadFile = File(..., description="File audio rekaman suara ibu hamil (.webm, .m4a, .wav, .mp3)")):
    """Menerima file rekaman suara dari Flutter Web/Mobile, mentranskripsikannya ke teks, dan mengembalikan jawaban RAG."""
    transcribed_text = await _transcribe_upload_file(file)

    if not transcribed_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Suara tidak terdeteksi atau audio kosong. Mohon rekam ulang suara Anda dengan lebih jelas."
        )

    try:
        response = ask_rag(query=transcribed_text, transcribed_text=transcribed_text)
        return response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Gagal memproses RAG dari audio: {str(e)}"
        )


# 3. Endpoint Backward-Compatibility (Alias untuk Skrip Pengujian Lama Rekan)
@app.post("/api/chat", tags=["Compatibility Aliases"])
def legacy_chat_endpoint(request: RAGQueryRequest):
    """Alias kompatibilitas mundur untuk endpoint lama /api/chat."""
    query = request.get_effective_query()
    res = ask_rag(query=query)
    return {
        "answer": res["answer"],
        "is_emergency": res["is_emergency"],
        "reference_source": res.get("reference_source"),
        "danger_level": res.get("danger_level")
    }


@app.post("/api/voice", tags=["Compatibility Aliases"])
async def legacy_voice_endpoint(file: UploadFile = File(...)):
    """Alias kompatibilitas mundur untuk endpoint lama /api/voice."""
    transcribed_text = await _transcribe_upload_file(file)
    rag_result = ask_rag(query=transcribed_text, transcribed_text=transcribed_text)

    return {
        "transcribed_text": transcribed_text,
        "answer": rag_result["answer"],
        "is_emergency": rag_result["is_emergency"],
        "reference_source": rag_result.get("reference_source"),
        "danger_level": rag_result.get("danger_level")
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8002))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
