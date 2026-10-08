toko_rina_audit/
│
├── data/
│   ├── raw/                        # Tempat menyimpan file mentah (PDF pelanggan, PDF penjualan, CSV mutasi bank)
│   └── processed/                  # Arsip folder harian otomatis per tanggal (contoh: folder "2026-09-18")
│       └── 2026-09-18/             # Folder Arsip Harian 
│           ├── uang_masuk_brankas.json
│           ├── uang_keluar_laci.json
│           ├── mutasi_db_bank.json
│           ├── catatan_utang.json
│           └── hasil_matching.json
│
├── src/
│   ├── config.py                   # Pengaturan konfigurasi (ambang batas skor, jendela 30 hari, dll.)
│   ├── database.py                 # Koneksi database SQLAlchemy (PostgreSQL / SQLite)
│   ├── models/                     # Definisi tabel database (Customer, Alias, Sales, Receivable, Cash, dll.)
│   ├── schemas/                    # Validasi data Pydantic untuk API request & response
│   ├── services/                   # Logika bisnis utama (Backend Services)
│   │   ├── normalizer.py           # Modul normalisasi alamat baku & cluster
│   │   ├── identity_resolver.py    # Mesin pencocok nama pengirim -> alamat -> akun (RapidFuzz)
│   │   ├── master_manager.py       # Logika CRUD master pelanggan, alias, & singkatan cluster
│   │   ├── cash_service.py         # Logika hitung kas laci, brankas, & selisih kas
│   │   ├── transfer_service.py     # Pencocokan mutasi bank CR dengan jendela 30 hari
│   │   ├── receivable_service.py   # Pengelolaan piutang, status lunas/sebagian, & pelunasan
│   │   ├── ledger_service.py       # Pencatatan mutasi DB (uang keluar bank) & saldo berjalan
│   │   ├── archive_service.py      # Modul generator arsip folder harian (seperti folder "18")
│   │   └── anomaly_ml_service.py   # Modul Machine Learning, pencarian kandidat, & deteksi anomali
│   │
│   ├── routers/                    # Endpoint API FastAPI (Master, Upload, Review, Dashboard)
│   ├── templates/                  # Tampilan HTML web interaktif (Bootstrap/Tailwind)
│   ├── static/                     # File CSS, JavaScript, atau Gambar untuk UI
│   └── main.py                     # Titik masuk utama aplikasi FastAPI
│
├── tests/                          # Unit tests (Pytest untuk normalisasi, resolver, & matching)
├── alembic/                        # Migrasi database
├── .env                            # Konfigurasi Environment (DB URL, dll.)
├── requirements.txt                # Daftar pustaka Python yang digunakan
└── README.md                       # Dokumentasi project