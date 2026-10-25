import os
import gc
import re
import logging
from typing import Dict, Any, List, Tuple
from dotenv import load_dotenv

# Optimasi alokasi memori Windows untuk mencegah WinError 1455
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import torch
try:
    torch.set_num_threads(1)
except Exception:
    pass

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import google.generativeai as genai

from setup_db import build_vector_db
from danger_classifier import ObstetricDangerClassifier, DANGER_TAXONOMY

load_dotenv()
logger = logging.getLogger("rag-engine")

# 1. Setup ChromaDB Vector Store & HuggingFace Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
db = Chroma(persist_directory="chroma_db", embedding_function=embeddings)


def _needs_reindex(vector_db: Chroma) -> bool:
    """Mengecek apakah ChromaDB masih kosong atau belum menggunakan versi pemisahan halaman Buku KIA terbaru."""
    try:
        sample = vector_db.get(include=["documents", "metadatas"])
        docs = sample.get("documents") or []
        metas = sample.get("metadatas") or []
        if not docs or not metas:
            return True
        if (metas[0] or {}).get("kia_v") != 2:
            return True
        return False
    except Exception:
        return False


if _needs_reindex(db):
    print("[*] Membangun indeks ChromaDB baru dengan pemulihan font & pemisahan 2-page spread...")
    db = build_vector_db(embeddings=embeddings)

# 2. Setup Dual-LLM (Primary: Groq Qwen, Fallback: Google Gemini)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

# Primary Provider: ChatGroq
try:
    llm = ChatGroq(model_name="qwen/qwen3.8-27b", temperature=0.1, api_key=GROQ_API_KEY)
except Exception as e:
    logger.warning(f"Inisialisasi ChatGroq gagal: {e}")
    llm = None

# Fallback Provider: Google Gemini
gemini_model = None
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel("gemini-3.8-flash")
    except Exception as e:
        logger.warning(f"Inisialisasi Google Gemini fallback gagal: {e}")


# 3. Helper Rekonstruksi Halaman Buku KIA (Metode Okta)
def _merge_page_chunks(chunks: list[str]) -> str:
    """Menggabungkan kembali chunks dalam satu halaman tanpa duplikasi bagian yang overlap."""
    if not chunks:
        return ""
    merged = chunks[0]
    for nxt in chunks[1:]:
        overlap_found = False
        max_ov = min(len(merged), len(nxt), 120)
        for ov in range(max_ov, 15, -1):
            if merged.endswith(nxt[:ov]):
                merged += nxt[ov:]
                overlap_found = True
                break
        if not overlap_found:
            merged += "\n" + nxt
    return merged


def _retrieve_full_kia_pages(query: str, max_pages: int = 5) -> list[tuple[str, str]]:
    """Mencari halaman Buku KIA paling relevan menggunakan Hybrid Search (Vektor + Kata Kunci Level Halaman)
    lalu menyusun kembali teks utuh dari halaman Buku KIA tersebut agar konteks tidak terpotong."""
    vector_docs = db.similarity_search(query, k=8)

    q_lower = query.lower()
    expanded_terms = set(re.findall(r"[a-z0-9]{3,}", q_lower))
    stopwords = {
        "apa", "apakah", "bagaimana", "berapa", "kapan", "mengapa", "kenapa",
        "yang", "dan", "atau", "untuk", "pada", "dalam", "dari", "dengan",
        "ibu", "saat", "bisa", "boleh", "harus", "saja", "kali",
    }
    keywords = expanded_terms - stopwords

    bonus_phrases = []
    if any(w in q_lower for w in ("minimal", "berapa kali", "frekuensi", "jadwal")):
        bonus_phrases.append("paling sedikit")
    if "periksa" in q_lower or "pemeriksaan" in q_lower:
        bonus_phrases.append("periksa kehamilan")
    if "melahirkan" in q_lower or "persalinan" in q_lower:
        bonus_phrases.extend(["tanda awal", "proses melahirkan"])
    if "menyusui" in q_lower or "asi" in q_lower:
        bonus_phrases.extend(["air susu ibu", "manfaat menyusui", "menyimpan asi"])

    all_data = db.get(include=["documents", "metadatas"])
    all_docs = all_data.get("documents") or []
    all_metas = all_data.get("metadatas") or []

    # Kelompokkan seluruh chunk berdasarkan nomor halaman asli Buku KIA
    pages_chunks: dict[str, list[tuple[int, str]]] = {}
    for doc_text, meta in zip(all_docs, all_metas):
        if not doc_text or not meta:
            continue
        p_label = str(meta.get("page_label") or meta.get("page") or "Tidak diketahui")
        c_idx = int(meta.get("chunk_idx", 0))
        pages_chunks.setdefault(p_label, []).append((c_idx, doc_text))

    full_pages: dict[str, str] = {}
    for p_label, clist in pages_chunks.items():
        clist.sort(key=lambda x: x[0])
        full_pages[p_label] = _merge_page_chunks([t for _, t in clist])

    # Beri skor gabungan untuk setiap halaman Buku KIA
    page_scores: dict[str, float] = {}

    # 1. Skor dari pencarian vektor semantik
    for rank, vdoc in enumerate(vector_docs):
        p_label = str(vdoc.metadata.get("page_label") or vdoc.metadata.get("page") or "")
        if p_label:
            page_scores[p_label] = page_scores.get(p_label, 0.0) + max(4.0 - rank * 0.4, 1.0)

    # 2. Skor dari kecocokan kata kunci & frasa pada satu halaman utuh
    for p_label, full_text in full_pages.items():
        t_lower = full_text.lower()
        kw_hits = sum(1.5 for kw in keywords if kw in t_lower)
        phrase_hits = sum(4.0 for phrase in bonus_phrases if phrase in t_lower)
        if kw_hits > 0 or phrase_hits > 0:
            score = page_scores.get(p_label, 0.0) + kw_hits + phrase_hits
            # Penalti ringan untuk halaman formulir pemantauan/nakes jika bukan itu yang ditanyakan
            if (
                "diisi oleh tenaga kesehatan" in t_lower
                or "diisi oleh dokter" in t_lower
                or "lembar pemantauan" in t_lower
            ) and "pencatatan" not in q_lower:
                score -= 3.5
            page_scores[p_label] = score

    ranked_pages = sorted(page_scores.items(), key=lambda x: x[1], reverse=True)
    selected_labels = [p for p, _ in ranked_pages[:max_pages] if p in full_pages]

    # Urutkan halaman terpilih berdasarkan nomor halaman agar alur baca LLM runut
    def _page_sort_key(lbl: str) -> int:
        return int(lbl) if lbl.isdigit() else 999

    selected_labels.sort(key=_page_sort_key)
    return [(lbl, full_pages[lbl]) for lbl in selected_labels]


# 4. Prompt Engineering Medis Anti-Halusinasi
prompt_template = """Kamu adalah Asisten Bidan Prenava yang ramah, empatik, jelas, dan berbasis bukti ilmiah.
Jawablah pertanyaan Ibu HANYA berdasarkan konteks resmi Buku KIA Kemenkes RI berikut:

{context}

Pertanyaan: {question}

Aturan Jawaban:
1. Jawablah secara akurat, lengkap, dan terstruktur sesuai isi Buku KIA di atas.
2. Jika informasi sama sekali tidak ditemukan di dalam konteks di atas, katakan persis: "Informasi tersebut belum tercantum dalam panduan ringkas Buku KIA ini, silakan konsultasi ke Bidan atau Dokter kandungan Anda."
3. DILARANG menebak, mengarang, atau meresepkan dosis obat keras/antibiotik tanpa resep dokter.
4. Wajib sebutkan nomor halaman Buku KIA yang menjadi rujukan di akhir jawaban (misal: "Rujukan: Buku KIA Kemenkes RI Halaman X").

Jawaban:
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])


# 5. Core RAG Inference Function (Terintegrasi Prenava & Backward Compatible)
def ask_rag(query: str, transcribed_text: str = None) -> Dict[str, Any]:
    effective_text = (transcribed_text or query or "").strip()
    if not effective_text:
        return {
            "status": "empty_query",
            "query": "",
            "answer": "Silakan masukkan pertanyaan seputar kesehatan ibu hamil dan anak.",
            "reference_source": "Buku KIA Kemenkes RI",
            "is_emergency": False,
            "danger_level": "NORMAL",
            "danger_keywords": [],
            "recommended_action": "Konsultasikan dengan Bidan pada jadwal ANC berikutnya.",
            "provider_used": "none",
            "retrieved_contexts": []
        }

    # A. Deteksi Tanda Bahaya Obstetri (Circuit Breaker)
    danger_eval = ObstetricDangerClassifier.evaluate(effective_text)
    if danger_eval["is_emergency"]:
        primary_cat = danger_eval["detected_categories"][0]
        emergency_answer = (
            f"🚨 PERHATIAN MEDIS SEGERA (TANDA BAHAYA KEHAMILAN) 🚨\n\n"
            f"Keluhan yang Ibu sampaikan terindikasi sebagai '{primary_cat['label']}'.\n\n"
            f"TINDAKAN MEDIS WAJIB DARI BUKU KIA KEMENKES RI:\n"
            f"• {primary_cat['recommended_action']}\n"
            f"• Jangan menunda waktu dan jangan mencoba mengobati sendiri di rumah.\n"
            f"• Mintalah keluarga untuk segera mendampingi menuju fasilitas kesehatan terdekat (Bidan/Puskesmas/IGD RS)."
        )
        return {
            "status": "emergency_alert",
            "query": query,
            "transcribed_text": transcribed_text,
            "answer": emergency_answer,
            "reference_source": "Buku KIA Kemenkes RI - Lembar Tanda Bahaya Kehamilan",
            "is_emergency": True,
            "danger_level": "EMERGENCY",
            "danger_keywords": danger_eval["matched_keywords"],
            "recommended_action": primary_cat["recommended_action"],
            "provider_used": "circuit_breaker",
            "retrieved_contexts": []
        }

    # B. Semantic & Hybrid Page-Level Retrieval (Metode Okta)
    matched_pages = _retrieve_full_kia_pages(effective_text, max_pages=5)

    context = "\n\n========================================\n\n".join(
        [
            f"[Sumber: Buku KIA Halaman {page_label}]\n{page_text}"
            for page_label, page_text in matched_pages
        ]
    )

    page_labels = [lbl for lbl, _ in matched_pages if lbl and lbl != "Tidak diketahui"]
    ref_source = f"Buku KIA Kemenkes RI - Halaman {', '.join(page_labels)}" if page_labels else "Buku KIA Kemenkes RI"

    # C. Dual-LLM Generation (Groq Primary -> Gemini Fallback)
    provider_used = "groq"
    answer_text = ""

    try:
        if llm is None:
            raise RuntimeError("ChatGroq belum terinisialisasi.")
        chain = prompt | llm
        response = chain.invoke({"context": context, "question": effective_text})
        answer_text = response.content if hasattr(response, "content") else str(response)
    except Exception as groq_err:
        logger.warning(f"[!] Groq primary error ({groq_err}), beralih ke Google Gemini fallback...")
        if gemini_model:
            try:
                full_prompt = prompt_template.format(context=context, question=effective_text)
                gemini_resp = gemini_model.generate_content(
                    full_prompt,
                    generation_config={"temperature": 0.1, "max_output_tokens": 800}
                )
                answer_text = gemini_resp.text
                provider_used = "gemini"
            except Exception as gem_err:
                logger.error(f"[X] Gemini fallback juga error: {gem_err}")
                answer_text = (
                    "Mohon maaf, sistem AI Bidan sedang mengalami gangguan koneksi ke server LLM. "
                    "Namun berdasarkan pencarian Buku KIA, Anda dapat merujuk ke " + ref_source
                )
                provider_used = "retrieval_only"
        else:
            answer_text = f"Terjadi kendala pada layanan AI: {str(groq_err)}"

    # Bersihkan tag pemikiran <think> jika ada
    answer_text = re.sub(r"<think>.*?</think>", "", answer_text, flags=re.DOTALL).strip()

    # Susun potongan konteks untuk transparansi frontend / evaluasi Ragas
    retrieved_contexts = [
        {
            "page": int(lbl) if lbl.isdigit() else 0,
            "source": f"Buku KIA Kemenkes RI - Halaman {lbl}",
            "similarity": 0.90,
            "danger_level": "NORMAL",
            "text_snippet": (page_text[:160] + "...") if len(page_text) > 160 else page_text
        }
        for lbl, page_text in matched_pages
    ]

    return {
        "status": "success",
        "query": query,
        "transcribed_text": transcribed_text,
        "answer": answer_text,
        "reference_source": ref_source,
        "is_emergency": False,
        "danger_level": "NORMAL",
        "danger_keywords": [],
        "recommended_action": "Terapkan panduan pola hidup sehat dan konsultasikan dengan Bidan pada jadwal ANC berikutnya.",
        "provider_used": provider_used,
        "retrieved_contexts": retrieved_contexts
    }
