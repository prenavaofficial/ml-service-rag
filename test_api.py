import io
import math
import wave
import struct
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def generate_test_wav(duration: float = 1.0, freq: float = 440.0) -> bytes:
    """Menghasilkan audio WAV sintetis sederhana untuk pengujian upload."""
    sample_rate = 16000
    num_samples = int(sample_rate * duration)
    wav_io = io.BytesIO()
    with wave.open(wav_io, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        for i in range(num_samples):
            val = int(32767.0 * 0.5 * math.sin(2.0 * math.pi * freq * i / sample_rate))
            wav_file.writeframes(struct.pack('<h', val))
    return wav_io.getvalue()


def test_root_endpoint():
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "online"
    assert data["port"] == 8002


def test_health_check_endpoint():
    resp = client.get("/rag/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "vector_store" in data


def test_danger_catalog_endpoint():
    resp = client.get("/rag/danger-catalog")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_categories"] == 6
    assert len(data["catalog"]) == 6


def test_ask_text_normal():
    resp = client.post(
        "/rag/ask-text",
        json={"query": "Berapa kali minimal pemeriksaan kehamilan menurut Buku KIA?"}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["is_emergency"] is False
    assert "Buku KIA" in data["reference_source"]
    assert len(data["answer"]) > 10


def test_ask_text_circuit_breaker():
    resp = client.post(
        "/rag/ask-text",
        json={"query": "Tolong bidan, dari tadi keluar darah segar dan gumpalan dari jalan lahir!"}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "emergency_alert"
    assert data["is_emergency"] is True
    assert data["danger_level"] == "EMERGENCY"
    assert len(data["danger_keywords"]) > 0


def test_legacy_chat_alias():
    resp = client.post(
        "/api/chat",
        json={"text": "Apa saja makanan bergizi untuk ibu hamil?"}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    assert "is_emergency" in data


if __name__ == "__main__":
    print("[*] Menjalankan seluruh pengujian endpoint FastAPI...")
    test_root_endpoint()
    print("[V] Root Endpoint: PASS")
    test_health_check_endpoint()
    print("[V] Health Check Endpoint: PASS")
    test_danger_catalog_endpoint()
    print("[V] Danger Catalog Endpoint: PASS")
    test_ask_text_normal()
    print("[V] Ask Text (Normal): PASS")
    test_ask_text_circuit_breaker()
    print("[V] Ask Text (Circuit Breaker): PASS")
    test_legacy_chat_alias()
    print("[V] Legacy Chat Alias: PASS")
    print("\n[V] SELURUH PENGUJIAN API GATEWAY PASS DENGAN SEMPURNA!")
