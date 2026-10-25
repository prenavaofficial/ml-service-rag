import os
import re
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from setup_db import build_vector_db

load_dotenv()

# 1. Setup Database & LLM
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
    db = build_vector_db(embeddings=embeddings)

llm = ChatGroq(model_name="qwen/qwen3.8-27b", temperature=0)

# 2. Emergency Danger Rule Classifier
DANGER_PATTERNS = [
    r"\bp[er]ndarahan\b",
    r"\bperdarahan\b",
    r"\bflek\s+banyak\b",
    r"\bkeluar\s+darah\b",
    r"\bketuban\s+(pecah|rembes|keluar|hijau|berbau)\b",
    r"\bair\s+ketuban\b",
    r"\b(kejang|pingsan|tidak\s+sadar)\b",
    r"\b(nyeri|sakit|kram)\s+perut\s+hebat\b",
    r"\bgerak(an)?\s+(bayi|janin)\s+(hilang|berkurang|tidak\s+ada|lemah)\b",
    r"\b(bayi|janin)\s+tidak\s+bergerak\b",
    r"\b(demam|panas)\s+tinggi\b",
    r"\bmuntah\s+(terus|hebat)\b",
    r"\btidak\s+mau\s+makan\b",
    r"\b(sakit\s+kepala|pusing)\s+(hebat|berat)\b",
    r"\bpandangan\s+kabur\b",
    r"\bbengkak\s+(di\s+)?(wajah|muka|tangan)\b",
]
DANGER_REGEX = re.compile("|".join(DANGER_PATTERNS), re.IGNORECASE)


def check_emergency(text: str) -> bool:
    if not text:
        return False
    return bool(DANGER_REGEX.search(text))


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

    # 1. Skor dari pencarian vektor semantik (urutan teratas mendapat bobot lebih besar)
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
            # Penalti ringan untuk halaman formulir kosong nakes/pemantauan jika bukan itu yang ditanyakan
            if (
                "diisi oleh tenaga kesehatan" in t_lower
                or "diisi oleh dokter" in t_lower
                or "lembar pemantauan" in t_lower
            ) and "pencatatan" not in q_lower:
                score -= 3.5
            page_scores[p_label] = score

    ranked_pages = sorted(page_scores.items(), key=lambda x: x[1], reverse=True)
    selected_labels = [p for p, _ in ranked_pages[:max_pages] if p in full_pages]

    # Urutkan halaman terpilih berdasarkan nomor halaman agar alur baca LLM berurutan (misal Trimester 1 -> 2 -> 3)
    def _page_sort_key(lbl: str) -> int:
        return int(lbl) if lbl.isdigit() else 999

    selected_labels.sort(key=_page_sort_key)
    return [(lbl, full_pages[lbl]) for lbl in selected_labels]


# 3. Prompt Engineering Anti-Halusinasi (DITAMBAH INSTRUKSI HALAMAN BUKU KIA ASLI)
prompt_template = """Kamu adalah Asisten Bidan Prenava yang ramah, jelas, dan berbasis bukti ilmiah.
Jawablah pertanyaan Ibu HANYA berdasarkan konteks Buku KIA Kemenkes RI berikut:

{context}

Pertanyaan: {question}

Aturan Jawaban:
1. Jawablah secara akurat, lengkap, dan terstruktur sesuai isi Buku KIA di atas.
2. Jika informasi sama sekali tidak ditemukan di dalam konteks di atas, katakan persis: "Informasi tersebut belum tercantum dalam panduan resmi Buku KIA, silakan konsultasi ke Bidan."
3. DILARANG menebak atau memberikan dosis obat keras.
4. Wajib sebutkan nomor halaman Buku KIA yang menjadi rujukan di akhir jawaban (hanya sebutkan nomor halaman yang benar-benar memuat informasi jawaban tersebut).

Jawaban:
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])


def ask_rag(query: str):
    query = (query or "").strip()
    if not query:
        return {
            "answer": "Silakan masukkan pertanyaan seputar kesehatan ibu dan anak.",
            "is_emergency": False,
        }

    is_emergency = check_emergency(query)
    if is_emergency:
        return {
            "answer": (
                "PERHATIAN: Gejala yang Ibu alami merupakan TANDA BAHAYA KEHAMILAN. "
                "SEGERA MENUJU KE KLINIK BIDAN, PUSKESMAS, ATAU IGD RUMAH SAKIT TERDEKAT!"
            ),
            "is_emergency": True,
        }

    matched_pages = _retrieve_full_kia_pages(query, max_pages=5)

    context = "\n\n========================================\n\n".join(
        [
            f"[Sumber: Buku KIA Halaman {page_label}]\n{page_text}"
            for page_label, page_text in matched_pages
        ]
    )

    chain = prompt | llm
    response = chain.invoke({"context": context, "question": query})

    # Bersihkan tag <think>...</think> jika model menyertakannya
    answer_text = re.sub(r"<think>.*?</think>", "", response.content, flags=re.DOTALL).strip()

    return {
        "answer": answer_text,
        "is_emergency": False,
    }
