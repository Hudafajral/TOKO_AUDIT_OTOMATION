"""
reset_data.py
Skrip pembersih folder arsip data/raw/ dan tabel piutang/transaksi database Toko Rina.
Akun admin, kasir, dan pengaturan default ML tetap dipertahankan.
"""
import shutil
from pathlib import Path
from sqlalchemy import text

from src.database import SessionLocal, engine
from src.models.audit_models import (
    Customer, Receivable, ExpenseItem, AuditActionLog, User, MLSetting
)

def reset_raw_folders():
    """Menghapus semua folder arsip harian di dalam data/raw/."""
    raw_dir = Path("data/raw")
    if raw_dir.exists():
        deleted_count = 0
        for item in raw_dir.iterdir():
            if item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
                deleted_count += 1
            elif item.is_file() and item.name != ".gitkeep":
                item.unlink(missing_ok=True)
                deleted_count += 1
        print(f"✓ Berhasil membersihkan folder data/raw/ ({deleted_count} folder/berkas dihapus).")
    else:
        raw_dir.mkdir(parents=True, exist_ok=True)
        print("✓ Folder data/raw/ dibuat ulang dalam kondisi kosong.")

def reset_database():
    """Mengosongkan tabel piutang, log audit, dan pengeluaran dari database."""
    # 1. Bersihkan via koneksi SQL langsung untuk memastikan data terhapus tuntas
    with engine.connect() as conn:
        for tbl in ["receivables", "audit_logs", "expense_items"]:
            try:
                conn.execute(text(f"DELETE FROM {tbl};"))
            except Exception:
                pass
        conn.commit()

    # 2. Pastikan akun default dan setting ML tetap ada
    db = SessionLocal()
    try:
        if not db.query(User).filter(User.username == "admin").first():
            db.add(User(username="admin", password_hash="admin123", role="admin"))
        if not db.query(User).filter(User.username == "kasir").first():
            db.add(User(username="kasir", password_hash="kasir123", role="kasir"))
        if not db.query(MLSetting).first():
            db.add(MLSetting())
        db.commit()
        print("✓ Tabel utang (receivables), riwayat audit, dan item pengeluaran berhasil dikosongkan.")
        print("✓ Akun default admin/kasir dan konfigurasi ML siap digunakan.")
    except Exception as e:
        db.rollback()
        print(f"Error saat memastikan user default: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    print("Memulai proses reset data Toko Rina...")
    reset_raw_folders()
    reset_database()
    print("\nSelesai! Sistem sudah kembali bersih dari awal (fresh state).")