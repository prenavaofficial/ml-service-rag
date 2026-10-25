import re
from typing import Dict, Any, List

# Taksonomi Tanda Bahaya Obstetri Resmi Buku KIA Kemenkes RI
DANGER_TAXONOMY = {
    "PERDARAHAN_ANTEPARTUM": {
        "label": "Perdarahan Pervaginam / Jalan Lahir",
        "keywords": [
            r"\bperdarahan\b", r"\bpendarahan\b", r"\bflek\s+darah\b", r"\bdarah\s+segar\b",
            r"\bkeluar\s+darah\b", r"\bgumpalan\s+darah\b", r"\bbercak\s+darah\s+banyak\b"
        ],
        "action": "SEGERA MENUJU KE KLINIK BIDAN ATAU IGD RUMAH SAKIT TERDEKAT! Berbaring miring ke kiri, jangan meminum obat apa pun tanpa resep dokter."
    },
    "PREEKLAMSIA_BERAT": {
        "label": "Kecurigaan Preeklamsia Berat / Hipertensi Gestasional",
        "keywords": [
            r"\bpandangan\s+kabur\b", r"\bberkunang-kunang\b", r"\bpusing\s+hebat\b",
            r"\bsakit\s+kepala\s+hebat\b", r"\bnyeri\s+ulu\s+hati\b", r"\btensi\s+(melonjak|tinggi)\b",
            r"\bbengkak\s+mendadak\b", r"\bkejang\b"
        ],
        "action": "BAHAYA PREEKLAMSIA! Segera periksakan tekanan darah dan urine ke Puskesmas/RS untuk mencegah terjadinya kejang eklamsia."
    },
    "KETUBAN_PECAH_DINI": {
        "label": "Kecurigaan Ketuban Pecah Dini (KPD)",
        "keywords": [
            r"\bketuban\s+pecah\b", r"\bcairan\s+ketuban\b", r"\bair\s+ketuban\b",
            r"\bbau\s+amis\b", r"\bcairan\s+merembes\b", r"\bair\s+merembes\s+dari\s+jalan\s+lahir\b"
        ],
        "action": "SEGERA KE FASILITAS KESEHATAN! Gunakan pembalut bersih, jangan memasukkan benda apa pun ke jalan lahir, dan segera ke Bidan/RS."
    },
    "GAWAT_JANIN": {
        "label": "Kecurigaan Gawat Janin (Fetal Distress)",
        "keywords": [
            r"\b(janin|bayi)\s+(tidak|kurang)\s+bergerak\b", r"\bgerakan?\s+janin\s+berhenti\b",
            r"\btidak\s+ada\s+tendangan\b", r"\btidak\s+bergerak\s+sama\s+sekali\b"
        ],
        "action": "SEGERA KE IGD/BIDAN UNTUK PEMERIKSAAN DENYUT JANTUNG JANIN (CTG/USG)! Rangsang janin dengan minum air dingin/manis lalu berbaring miring ke kiri."
    },
    "INFEKSI_AKUT_DEMAM": {
        "label": "Demam Tinggi / Gejala Infeksi Akut",
        "keywords": [
            r"\bdemam\s+tinggi\b", r"\bmenggigil\s+hebat\b", r"\bpanas\s+tinggi\b",
            r"\bkejang-kejang\b"
        ],
        "action": "BAHAYA INFEKSI MATERNAL! Segera bawa ibu ke Puskesmas/Rumah Sakit untuk evaluasi sumber infeksi dan pencegahan komplikasi."
    },
    "NYERI_PERUT_AKUT": {
        "label": "Nyeri Abdomen Hebat / Kontraksi Prematur",
        "keywords": [
            r"\bnyeri\s+perut\s+hebat\b", r"\bperut\s+kencang\s+terus\b",
            r"\bmulas\s+teratur\b", r"\bperut\s+kaku\s+seperti\s+papan\b"
        ],
        "action": "SEGERA PERIKSAKAN DIRI! Kontraksi terus-menerus atau nyeri perut hebat memerlukan evaluasi plasenta dan pencegahan persalinan prematur."
    }
}


class ObstetricDangerClassifier:
    """Engine pendeteksi tanda bahaya kehamilan dan pemicu Circuit Breaker."""

    @classmethod
    def evaluate(cls, text: str) -> Dict[str, Any]:
        text_clean = (text or "").lower()
        detected_categories: List[Dict[str, Any]] = []
        all_matched_keywords: List[str] = []

        for category_key, info in DANGER_TAXONOMY.items():
            matched_in_category = []
            for pattern in info["keywords"]:
                found = re.findall(pattern, text_clean)
                if found:
                    if isinstance(found[0], tuple):
                        matched_in_category.append(" ".join(found[0]))
                    else:
                        matched_in_category.append(found[0])

            if matched_in_category:
                unique_matches = list(set(matched_in_category))
                all_matched_keywords.extend(unique_matches)
                detected_categories.append({
                    "code": category_key,
                    "label": info["label"],
                    "matched_keywords": unique_matches,
                    "recommended_action": info["action"]
                })

        is_emergency = len(detected_categories) > 0

        if is_emergency:
            primary_cat = detected_categories[0]
            alert_message = (
                f"PERHATIAN KHUSUS (TANDA BAHAYA OBSTETRI): Gejala yang Ibu alami terindikasi sebagai "
                f"'{primary_cat['label']}' yang memerlukan tindakan medis segera!"
            )
            action_instruction = primary_cat["recommended_action"]
        else:
            alert_message = "Kondisi tidak terdeteksi adanya tanda bahaya obstetri akut."
            action_instruction = "Lanjutkan pemantauan kehamilan dan konsultasi rutin dengan Bidan."

        return {
            "is_emergency": is_emergency,
            "danger_level": "EMERGENCY" if is_emergency else "NORMAL",
            "alert_message": alert_message,
            "recommended_action": action_instruction,
            "detected_categories": detected_categories,
            "matched_keywords": list(set(all_matched_keywords))
        }
