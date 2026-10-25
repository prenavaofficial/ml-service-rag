from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from ingest_docs import load_and_chunk_pdf

# Konfigurasi Path
pdf_path = "data/buku_kia.pdf"
db_path = "chroma_db"


def build_vector_db(pdf_file: str = pdf_path, persist_dir: str = db_path, embeddings=None) -> Chroma:
    print("1. Memuat, memperbaiki font PDF, dan memotong dokumen Buku KIA...")
    chunks = load_and_chunk_pdf(pdf_file, chunk_size=500, chunk_overlap=80)
    print(f"   Total chunks bersih siap disimpan: {len(chunks)}")

    if embeddings is None:
        print("2. Menyiapkan model embedding...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

    print("3. Menyimpan vektor ke database lokal (ChromaDB)...")
    db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)

    # Hapus data lama terlebih dahulu agar tidak terjadi duplikasi saat dijalankan ulang
    existing_ids = db.get().get("ids", [])
    if existing_ids:
        print(f"   Menghapus {len(existing_ids)} chunks lama dari database...")
        for i in range(0, len(existing_ids), 500):
            db.delete(ids=existing_ids[i : i + 500])

    db.add_documents(chunks)
    print("Database berhasil diperbarui dan disimpan!")
    return db


if __name__ == "__main__":
    db = build_vector_db()

    print("\n4. Menguji Pencarian Semantik...")
    query = "Apa tanda awal proses melahirkan?"
    docs = db.similarity_search(query, k=2)

    for i, doc in enumerate(docs):
        page_label = doc.metadata.get("page_label") or doc.metadata.get("page")
        print(f"\n--- Hasil {i+1} (Halaman PDF: {page_label}) ---")
        print(doc.page_content)
