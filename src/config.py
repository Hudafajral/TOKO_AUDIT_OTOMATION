import os
from dotenv import load_dotenv

# Memuat variabel lingkungan dari file .env
load_dotenv()

class Settings:
    # Konfigurasi Database (Default ke SQLite untuk kemudahan dev lokal, bisa diganti PostgreSQL)
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./toko_rina_audit.db")
    
    # Pengaturan Bisnis & Matching
    TRANSFER_MATCHING_WINDOW_DAYS: int = 30  # Jendela waktu toleransi pencocokan transfer (30 hari)
    FUZZY_MATCH_THRESHOLD: int = 75          # Ambang batas minimum skor kemiripan teks (RapidFuzz)
    AUTO_MATCH_CONFIDENCE: int = 90          # Ambang batas skor untuk auto-match tanpa review manual

settings = Settings()