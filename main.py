import os
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from groq import Groq
from rag_engine import ask_rag
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Prenava Medical RAG API")
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


# Skema data untuk chat teks
class QueryRequest(BaseModel):
    text: str


# 1. Endpoint Teks (Chat Biasa)
@app.post("/api/chat")
def chat_endpoint(request: QueryRequest):
    return ask_rag(request.text)


# 2. Endpoint Suara (Voice-Enabled)
@app.post("/api/voice")
async def voice_endpoint(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
    temp_file_path = None

    try:
        audio_bytes = await file.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="File audio kosong.")

        # Simpan file audio sementara dengan nama unik yang aman
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
            temp_file.write(audio_bytes)
            temp_file_path = temp_file.name

        # Transkripsi audio menjadi teks menggunakan Groq Whisper (Bahasa Indonesia)
        with open(temp_file_path, "rb") as audio_file:
            transcription = groq_client.audio.transcriptions.create(
                file=(os.path.basename(temp_file_path), audio_file.read()),
                model="whisper-large-v3-turbo",
                language="id",
                response_format="json",
            )

        user_text = transcription.text
        rag_result = ask_rag(user_text)

        return {
            "transcribed_text": user_text,
            "answer": rag_result["answer"],
            "is_emergency": rag_result["is_emergency"],
        }
    finally:
        # Pastikan file sementara selalu dihapus meskipun terjadi error
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except OSError:
                pass


# 3. Cek Status Server
@app.get("/")
def root():
    return {"status": "Prenava ML Service is Running Active"}
