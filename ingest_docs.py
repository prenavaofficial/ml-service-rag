import io
import re
import pypdf
from pypdf.generic import DecodedStreamObject, NameObject
from fontTools.ttLib import TTFont
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

pdf_path = "data/buku_kia.pdf"


def _repair_pdf_fonts(reader: pypdf.PdfReader) -> None:
    """Memperbaiki font Type0 (Identity-H) yang tidak memiliki tabel /ToUnicode
    pada PDF hasil ekspor Adobe Illustrator dengan mencocokkan koordinat glyph TrueType."""
    global_cmap = {}
    any_font_cmap = {}

    # 1. Kumpulkan peta bentuk glyph -> kode karakter Unicode dari font yang memiliki cmap
    for page in reader.pages:
        res = page.get("/Resources")
        if not res or "/Font" not in res:
            continue
        for _, fobj in res["/Font"].items():
            f = fobj.get_object()
            base_font = str(f.get("/BaseFont", "")).split("+")[-1]
            fd = None
            if "/FontDescriptor" in f:
                fd = f["/FontDescriptor"].get_object()
            elif "/DescendantFonts" in f:
                df = f["/DescendantFonts"][0].get_object()
                if "/FontDescriptor" in df:
                    fd = df["/FontDescriptor"].get_object()
            if not fd or "/FontFile2" not in fd:
                continue
            try:
                tt = TTFont(io.BytesIO(fd["/FontFile2"].get_data()))
                if "cmap" not in tt or "glyf" not in tt:
                    continue
                cmap = tt.getBestCmap()
                if not cmap:
                    continue
                glyf = tt["glyf"]
                for code, gname in cmap.items():
                    if gname in glyf and 32 < code < 65535:
                        coords, end_pts, _ = glyf[gname].getCoordinates(glyf)
                        if len(coords) > 0:
                            key = (tuple(coords), tuple(end_pts))
                            global_cmap[(base_font, key)] = code
                            any_font_cmap[key] = code
            except Exception:
                continue

    # 2. Sisipkan stream /ToUnicode ke seluruh font Type0 yang belum memilikinya
    for page in reader.pages:
        res = page.get("/Resources")
        if not res or "/Font" not in res:
            continue
        for _, fobj in res["/Font"].items():
            f = fobj.get_object()
            if "/DescendantFonts" in f and "/ToUnicode" not in f:
                base_font = str(f.get("/BaseFont", "")).split("+")[-1]
                df = f["/DescendantFonts"][0].get_object()
                if "/FontDescriptor" not in df:
                    continue
                fd = df["/FontDescriptor"].get_object()
                if "/FontFile2" not in fd:
                    continue
                try:
                    tt = TTFont(io.BytesIO(fd["/FontFile2"].get_data()))
                    if "glyf" not in tt:
                        continue
                    glyf = tt["glyf"]
                    pairs = []
                    for cid, gname in enumerate(tt.getGlyphOrder()):
                        if cid == 0:
                            continue
                        coords, end_pts, _ = glyf[gname].getCoordinates(glyf)
                        if len(coords) == 0:
                            code = 32
                        else:
                            key = (tuple(coords), tuple(end_pts))
                            code = global_cmap.get((base_font, key)) or any_font_cmap.get(key) or 32
                        pairs.append(f"<{cid:04X}> <{code:04X}>")

                    cmap_lines = [
                        "/CIDInit /ProcSet findresource begin",
                        "12 dict begin",
                        "begincmap",
                        "/CMapName /Custom def",
                        "/CMapType 2 def",
                        "1 begincodespacerange",
                        "<0000> <FFFF>",
                        "endcodespacerange",
                    ]
                    for i in range(0, len(pairs), 100):
                        batch = pairs[i : i + 100]
                        cmap_lines.append(f"{len(batch)} beginbfchar")
                        cmap_lines.extend(batch)
                        cmap_lines.append("endbfchar")
                    cmap_lines.extend(
                        ["endcmap", "CMapName currentdict /CMap defineresource pop", "end", "end"]
                    )
                    stream = DecodedStreamObject()
                    stream.set_data("\n".join(cmap_lines).encode("ascii"))
                    f[NameObject("/ToUnicode")] = stream
                except Exception:
                    continue


def _clean_page_text(raw_text: str) -> str:
    """Membersihkan karakter kontrol, memperbaiki spasi judul, dan menghapus baris nomor halaman/tabel kosong."""
    clean_text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\ufffd]", " ", raw_text or "")
    clean_text = (
        clean_text.replace("KEH AMIL AN", "KEHAMILAN")
        .replace("MEL AHIRKAN", "MELAHIRKAN")
        .replace("SETEL AH", "SETELAH")
        .replace("BUL AN", "BULAN")
        .replace("INFORM ASI", "INFORMASI")
        .replace("PENCATATA N", "PENCATATAN")
        .replace("TEN AGA", "TENAGA")
        .replace("KESEH ATA N", "KESEHATAN")
        .replace("ﬁ", "fi")
    )
    # Hapus baris yang hanya berisi angka nomor halaman ganda (misal '1918 19', '22 23') atau nomor tabel kosong
    clean_text = re.sub(r"^\s*\d{1,6}(?:\s+\d{1,6})*\s*$", "", clean_text, flags=re.MULTILINE)
    clean_text = re.sub(r"[ \t]+", " ", clean_text)
    clean_text = re.sub(r"\n{3,}", "\n\n", clean_text).strip()
    return clean_text


def load_and_chunk_pdf(
    path: str = pdf_path, chunk_size: int = 500, chunk_overlap: int = 80
) -> list[Document]:
    """Memuat PDF Buku KIA, memisahkan halaman kiri & kanan (2-page spread) sesuai
    nomor halaman asli Buku KIA (Halaman 1 s.d. 155), lalu membagi menjadi chunks."""
    reader = pypdf.PdfReader(path)
    _repair_pdf_fonts(reader)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    kia_pages: list[Document] = []
    for idx, page in enumerate(reader.pages):
        width = float(page.mediabox.width)

        # Lembar pertama adalah sampul tunggal (width ~565), lembar berikutnya 2 halaman berdampingan (width ~1064)
        if width < 800 or idx == 0:
            clean = _clean_page_text(page.extract_text() or "")
            if clean:
                kia_pages.append(
                    Document(
                        page_content=clean,
                        metadata={
                            "source": path,
                            "page": 0,
                            "page_label": "Sampul",
                            "pdf_sheet": idx + 1,
                        },
                    )
                )
            continue

        mid = width / 2.0
        left_parts: list[str] = []
        right_parts: list[str] = []
        last_y = {"L": None, "R": None}

        def _visitor(text, cm, tm, font_dict, font_size):
            if not text or not text.strip():
                return
            x = cm[0] * tm[4] + cm[2] * tm[5] + cm[4]
            y = cm[1] * tm[4] + cm[3] * tm[5] + cm[5]
            side = "L" if x < mid else "R"
            target = left_parts if side == "L" else right_parts
            if last_y[side] is not None and abs(y - last_y[side]) > 2.5:
                if target and not target[-1].endswith("\n"):
                    target.append("\n")
            last_y[side] = y
            target.append(text)

        page.extract_text(visitor_text=_visitor)

        if idx == 1:
            left_num, left_label = 0, "ii"
            right_num, right_label = 1, "1"
        else:
            left_num = 2 * idx - 2
            right_num = 2 * idx - 1
            left_label = str(left_num)
            right_label = str(right_num)

        left_clean = _clean_page_text("".join(left_parts))
        if left_clean:
            kia_pages.append(
                Document(
                    page_content=left_clean,
                    metadata={
                        "source": path,
                        "page": left_num,
                        "page_label": left_label,
                        "pdf_sheet": idx + 1,
                    },
                )
            )

        right_clean = _clean_page_text("".join(right_parts))
        if right_clean:
            kia_pages.append(
                Document(
                    page_content=right_clean,
                    metadata={
                        "source": path,
                        "page": right_num,
                        "page_label": right_label,
                        "pdf_sheet": idx + 1,
                    },
                )
            )

    chunks: list[Document] = []
    for pdoc in kia_pages:
        sub_chunks = text_splitter.split_documents([pdoc])
        for chunk_idx, sc in enumerate(sub_chunks):
            sc.metadata["chunk_idx"] = chunk_idx
            sc.metadata["kia_v"] = 2
            chunks.append(sc)

    return chunks


if __name__ == "__main__":
    print("Memuat, memperbaiki font, dan memisahkan halaman Buku KIA...")
    chunks = load_and_chunk_pdf(pdf_path)
    print(f"Berhasil membagi dokumen menjadi {len(chunks)} chunks bersih (Halaman asli Buku KIA).")
    print(f"Contoh metadata chunk pertama: {chunks[0].metadata}")
    print(f"Kutipan chunk pertama:\n{chunks[0].page_content[:200]}...")
