import re

def normalize_address(raw_text: str) -> str:
    """
    Fungsi untuk menormalisasi teks alamat mentah menjadi satu KUNCI ALAMAT BAKU.
    Alur:
    1. Ubah seluruh teks menjadi huruf kapital (uppercase).
    2. Ganti pemisah seperti garis miring (/), strip (-), koma (,), titik (.), dan garis bawah (_) dengan spasi.
    3. Hapus spasi berlebih di antara kata atau di awal/akhir teks.
    """
    if not raw_text:
        return ""
    
    # 1. Ubah ke uppercase
    text = raw_text.upper()
    
    # 2. Ganti tanda baca khusus dengan spasi
    text = re.sub(r'[/\\,\.\-_]', ' ', text)
    
    # 3. Rapikan spasi ganda menjadi spasi tunggal dan trim
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def normalize_cluster_abbreviation(cluster_code: str, cluster_map: dict) -> str:
    """
    Fungsi untuk menerjemahkan singkatan cluster (misal: 'PS' jadi 'PASADENA', 'MLD' jadi 'MALIBU').
    Jika singkatan tidak ditemukan di kamus, kembalikan teks aslinya.
    """
    clean_code = cluster_code.upper().strip()
    return cluster_map.get(clean_code, clean_code)