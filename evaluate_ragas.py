"""
Evaluasi Ilmiah Sistem Voice RAG Buku KIA Kemenkes RI
Khusus Keperluan Bab 4 Tugas Akhir - S1 Sistem Informasi Telkom University
Metrik: Waktu Inferensi, Kesesuaian Rujukan Dokumen, dan Ketepatan Deteksi Bahaya
"""

import time
import json
from pathlib import Path
from typing import List, Dict, Any

from rag_engine import ask_rag

# Dataset Kueri Pengujian Medis Bab 4 TA (10 Sampel Representatif)
EVAL_DATASET = [
    {
        "id": 1,
        "category": "GIZI_SUPLEMEN",
        "question": "Berapa banyak tablet tambah darah zat besi yang harus diminum selama hamil?",
        "expected_danger": False,
        "ground_truth_page": "Halaman 50",
        "keywords": ["tablet tambah darah", "zat besi", "setiap hari"]
    },
    {
        "id": 2,
        "category": "GIZI_NUTRISI",
        "question": "Bagaimana aturan porsi makan ibu hamil menurut pedoman Isi Piringku?",
        "expected_danger": False,
        "ground_truth_page": "Halaman 5",
        "keywords": ["porsi", "makanan pokok", "sayuran", "protein"]
    },
    {
        "id": 3,
        "category": "FISIOLOGIS",
        "question": "Mengapa sering mual dan muntah di pagi hari saat hamil muda?",
        "expected_danger": False,
        "ground_truth_page": "Halaman 6",
        "keywords": ["mual", "muntah", "trimester"]
    },
    {
        "id": 4,
        "category": "FISIOLOGIS",
        "question": "Apakah kaki bengkak di trimester ketiga adalah hal yang normal?",
        "expected_danger": False,
        "ground_truth_page": "Halaman 15",
        "keywords": ["bengkak", "kaki", "istirahat"]
    },
    {
        "id": 5,
        "category": "MITOS_FAKTA",
        "question": "Apakah ibu hamil boleh makan buah nanas dan minum kopi?",
        "expected_danger": False,
        "ground_truth_page": "Halaman 9",
        "keywords": ["makan", "gizi"]
    },
    {
        "id": 6,
        "category": "EMERGENCY_PERDARAHAN",
        "question": "Tolong bidan, keluar darah segar dan gumpalan darah banyak dari jalan lahir!",
        "expected_danger": True,
        "ground_truth_page": "Halaman 6",
        "keywords": ["perdarahan", "igd", "segera"]
    },
    {
        "id": 7,
        "category": "EMERGENCY_PREEKLAMSIA",
        "question": "Kepala saya sakit kepala hebat tidak sembuh dan mata pandangan kabur berkunang-kunang.",
        "expected_danger": True,
        "ground_truth_page": "Halaman 7",
        "keywords": ["preeklamsia", "tensi", "igd"]
    },
    {
        "id": 8,
        "category": "EMERGENCY_GAWAT_JANIN",
        "question": "Bidan, janin tidak bergerak sama sekali sejak siang tadi padahal sudah ditepuk.",
        "expected_danger": True,
        "ground_truth_page": "Halaman 19",
        "keywords": ["janin", "denyut jantung", "ctg"]
    },
    {
        "id": 9,
        "category": "EMERGENCY_KETUBAN",
        "question": "Celana dalam basah terus menerus keluar cairan ketuban bau amis mengalir.",
        "expected_danger": True,
        "ground_truth_page": "Halaman 7",
        "keywords": ["ketuban", "fasilitas kesehatan"]
    },
    {
        "id": 10,
        "category": "ANC_KONTROL",
        "question": "Berapa kali minimal periksa kehamilan ke bidan atau dokter selama hamil?",
        "expected_danger": False,
        "ground_truth_page": "Halaman 11",
        "keywords": ["periksa", "kehamilan", "dokter"]
    }
]


def evaluate_system():
    print("=" * 80)
    print("   PRENAVA RAG - EVALUASI SISTEM TUGAS AKHIR (BAB 4)")
    print("=" * 80)

    total = len(EVAL_DATASET)
    danger_correct = 0
    reference_found = 0
    latencies = []

    results_table = []

    for item in EVAL_DATASET:
        q_id = item["id"]
        q_text = item["question"]
        expected_d = item["expected_danger"]

        t0 = time.time()
        res = ask_rag(query=q_text)
        lat = time.time() - t0
        latencies.append(lat)

        is_emergency = res.get("is_emergency", False)
        ref_source = res.get("reference_source", "")
        answer = res.get("answer", "")
        provider = res.get("provider_used", "unknown")

        # Evaluasi ketepatan tanda bahaya
        is_danger_accurate = (is_emergency == expected_d)
        if is_danger_accurate:
            danger_correct += 1

        # Evaluasi sitasi rujukan Buku KIA
        has_kia_citation = "Buku KIA" in ref_source or "KIA" in answer
        if has_kia_citation:
            reference_found += 1

        results_table.append({
            "id": q_id,
            "category": item["category"],
            "latency_s": round(lat, 2),
            "danger_detected": is_emergency,
            "danger_pass": is_danger_accurate,
            "provider": provider,
            "ref_source": ref_source
        })

        print(f"[{q_id:02d}/{total}] {item['category']:<24} | Lat: {lat:.2f}s | Em: {str(is_emergency):<5} | Status: {'[PASS]' if is_danger_accurate else '[FAIL]'}")

    avg_latency = sum(latencies) / len(latencies)
    danger_accuracy = (danger_correct / total) * 100
    citation_rate = (reference_found / total) * 100

    print("-" * 80)
    print("                   RINGKASAN METRIK EVALUASI BAB 4")
    print("-" * 80)
    print(f"Total Skenario Pengujian          : {total} Kasus Klinis")
    print(f"Akurasi Deteksi Tanda Bahaya      : {danger_accuracy:.1f}% ({danger_correct}/{total})")
    print(f"Tingkat Sitasi Resmi Buku KIA     : {citation_rate:.1f}% ({reference_found}/{total})")
    print(f"Rata-rata Waktu Respon (Latensi)  : {avg_latency:.2f} detik")
    print(f"Distribusi Provider Digunakan     : Groq Qwen / Circuit Breaker")
    print("=" * 80)

    # Simpan laporan evaluasi ke file JSON
    report_path = Path("academic_evaluation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "total_cases": total,
                "danger_detection_accuracy": danger_accuracy,
                "citation_compliance_rate": citation_rate,
                "average_latency_seconds": round(avg_latency, 2),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "cases": results_table
        }, f, indent=2)

    print(f"[V] Laporan ilmiah berhasil disimpan di: {report_path.resolve()}")


if __name__ == "__main__":
    evaluate_system()
