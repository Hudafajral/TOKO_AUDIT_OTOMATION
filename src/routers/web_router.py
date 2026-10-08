"""
src/routers/web_router.py
API Controller Keuangan POS Toko Rina:
- Auto-migrasi skema database SQLite saat startup (termasuk status approval & lokasi IP)
- Login dengan verifikasi status persetujuan Admin
- Registrasi kasir baru dengan status PENDING dan deteksi IP/Lokasi
- Endpoint Approve / Reject user & penghitungan pending badge counter
- Dashboard Summary Kumulatif + Saldo Awal Baseline (Cash, Transfer, Utang)
- Run Audit Presisi dengan Sinkronisasi Database
- Perpustakaan Dokumen: Upload, List, Download, Toggle Status
- Audit Trail Terpisah & Pengaturan Saldo/ML Terpisah
"""
from __future__ import annotations

import io
import json
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.database import SessionLocal, engine, Base
from src.models.audit_models import (
    User, Customer, Receivable, AuditActionLog, MLSetting, UploadedDocument
)
from src.services.ingestion_service import parse_sales_report_pdf, parse_bank_mutation_csv
from src.services.receivable_service import (
    record_daily_credit_sales, get_all_debts_for_table, get_total_receivable_balance, settle_receivable
)
from src.services.transfer_service import reconcile_daily_bank_mutations
from src.services.anomaly_ml_service import evaluate_daily_audit_risks
from src.services.archive_service import save_daily_cash_files

router = APIRouter(tags=["Web App"])
BASE_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

DOCS_DIR = Path("data/documents")
DOCS_DIR.mkdir(parents=True, exist_ok=True)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def detect_client_info(request: Request) -> tuple[str, str]:
    """Mendeteksi IP asli dan lokasi geografis sederhana klien."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        client_ip = forwarded.split(",")[0].strip()
    else:
        client_ip = request.client.host if request.client else "127.0.0.1"

    if client_ip in ["127.0.0.1", "localhost", "::1"] or client_ip.startswith("192.168.") or client_ip.startswith("10."):
        return client_ip, "Lokal (Jaringan Toko Rina)"

    location_str = "Indonesia"
    try:
        # Coba ambil kota dari IP publik (timeout cepat 1 detik)
        req = urllib.request.Request(f"https://ipapi.co/{client_ip}/json/", headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=1.0) as resp:
            data = json.loads(resp.read().decode())
            city = data.get("city")
            region = data.get("region")
            if city:
                location_str = f"{city}, {region}" if region else city
    except Exception:
        pass

    return client_ip, location_str


@router.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)

    # Migrasi Kolom Otomatis di SQLite
    with engine.connect() as conn:
        cols_to_add = [
            ("ml_settings", "initial_cash_balance", "FLOAT DEFAULT 0.0"),
            ("ml_settings", "initial_transfer_balance", "FLOAT DEFAULT 0.0"),
            ("ml_settings", "initial_debt_balance", "FLOAT DEFAULT 0.0"),
            ("users", "allowed_tabs", "TEXT DEFAULT '[\"homeDashboardTab\",\"auditFormTab\",\"docLibraryTab\",\"piutangTab\"]'"),
            ("users", "status_approval", "VARCHAR(20) DEFAULT 'APPROVED'"),
            ("users", "registered_ip", "VARCHAR(50) DEFAULT '127.0.0.1'"),
            ("users", "registered_location", "VARCHAR(100) DEFAULT 'Lokal / Internal'"),
            ("audit_action_logs", "client_location", "VARCHAR(100) DEFAULT 'Lokal / Internal'")
        ]
        for tbl, col_name, col_type in cols_to_add:
            try:
                conn.execute(text(f"ALTER TABLE {tbl} ADD COLUMN {col_name} {col_type};"))
                conn.commit()
            except Exception:
                pass

    db = SessionLocal()
    try:
        admin_u = db.query(User).filter(User.username == "admin").first()
        if not admin_u:
            db.add(User(
                username="admin", 
                password_hash="admin123", 
                role="admin", 
                status_approval="APPROVED",
                allowed_tabs='["*"]',
                registered_ip="127.0.0.1",
                registered_location="Sistem Utama"
            ))
        else:
            admin_u.role = "admin"
            admin_u.status_approval = "APPROVED"
            admin_u.allowed_tabs = '["*"]'
        if not db.query(MLSetting).first():
            db.add(MLSetting())
        db.commit()
    finally:
        db.close()


@router.get("/", response_class=HTMLResponse)
def index_page(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={})


# ================= AUTENTIKASI DENGAN APPROVAL & LOKASI IP =================
class AuthPayload(BaseModel):
    username: str
    password: str


@router.get("/api/check-user-status/{username}")
def check_user_status(username: str, db: Session = Depends(get_db)):
    """Memeriksa apakah akun sudah terdaftar dan status approval-nya untuk animasi gembok."""
    uname = username.strip().lower()
    u = db.query(User).filter(User.username == uname).first()
    if not u:
        return {"exists": False, "status": "NOT_FOUND"}
    return {
        "exists": True, 
        "status": u.status_approval or "APPROVED",
        "role": u.role
    }


@router.post("/api/login")
def login(payload: AuthPayload, request: Request, db: Session = Depends(get_db)):
    uname = payload.username.strip().lower()
    u = db.query(User).filter(User.username == uname, User.password_hash == payload.password).first()
    if not u:
        raise HTTPException(status_code=401, detail="Username atau password salah!")

    # Cek status approval
    if u.role != "admin" and (u.status_approval or "").upper() == "PENDING":
        raise HTTPException(
            status_code=403, 
            detail="Akun Anda sedang menunggu persetujuan Admin Toko Rina. Hubungi admin untuk aktivasi!"
        )
    if u.role != "admin" and (u.status_approval or "").upper() == "REJECTED":
        raise HTTPException(
            status_code=403, 
            detail="Permintaan akses akun Anda ditolak oleh Admin."
        )

    client_ip, client_loc = detect_client_info(request)

    allowed = ["*"]
    if u.role != "admin":
        try:
            allowed = json.loads(u.allowed_tabs or '["homeDashboardTab","auditFormTab","docLibraryTab","piutangTab"]')
        except Exception:
            allowed = ["homeDashboardTab", "auditFormTab", "docLibraryTab", "piutangTab"]

    db.add(AuditActionLog(
        username=u.username,
        action_type="LOGIN",
        description=f"Pengguna '{u.username}' berhasil login ({client_loc})",
        client_ip=client_ip,
        client_location=client_loc
    ))
    db.commit()
    return {
        "status": "SUCCESS", 
        "username": u.username, 
        "role": u.role,
        "allowed_tabs": allowed,
        "client_ip": client_ip,
        "client_location": client_loc
    }


@router.post("/api/register-kasir")
def register_kasir(payload: AuthPayload, request: Request, db: Session = Depends(get_db)):
    uname = payload.username.strip().lower()
    pwd = payload.password.strip()
    if uname == "admin" or pwd == "admin123":
        raise HTTPException(status_code=400, detail="Username/Password kasir tidak boleh sama dengan kredensial Admin!")

    existing = db.query(User).filter(User.username == uname).first()
    if existing:
        raise HTTPException(status_code=400, detail="Username kasir sudah terdaftar!")

    client_ip, client_loc = detect_client_info(request)
    default_permissions = json.dumps(["homeDashboardTab", "auditFormTab", "docLibraryTab", "piutangTab"])

    new_user = User(
        username=uname, 
        password_hash=pwd, 
        role="kasir",
        status_approval="PENDING",  # WAJIB MENUNGGU PERSETUJUAN ADMIN
        registered_ip=client_ip,
        registered_location=client_loc,
        allowed_tabs=default_permissions
    )
    db.add(new_user)
    db.add(AuditActionLog(
        username="SYSTEM",
        action_type="USER_REGISTER",
        description=f"Kasir baru mendaftar: '{uname}' dari {client_ip} ({client_loc}). Menunggu persetujuan.",
        client_ip=client_ip,
        client_location=client_loc
    ))
    db.commit()
    return {
        "status": "SUCCESS", 
        "message": f"Pendaftaran berhasil! Akun '{uname}' sedang menunggu persetujuan (approval) dari Admin.",
        "client_ip": client_ip,
        "client_location": client_loc
    }


# ================= MANAJEMEN USER & APPROVAL (ADMIN ONLY) =================
@router.get("/api/admin/users")
def get_user_list(db: Session = Depends(get_db)):
    """Mengambil daftar seluruh pengguna, IP, lokasi, status approval, dan hak akses."""
    users = db.query(User).order_by(User.id.desc()).all()
    pending_count = db.query(User).filter(User.status_approval == "PENDING").count()
    out = []
    for u in users:
        tabs = []
        try:
            tabs = json.loads(u.allowed_tabs or "[]")
        except Exception:
            tabs = []
        out.append({
            "id": u.id,
            "username": u.username,
            "role": u.role,
            "status_approval": u.status_approval or "APPROVED",
            "registered_ip": u.registered_ip or "127.0.0.1",
            "registered_location": u.registered_location or "Lokal",
            "allowed_tabs": tabs,
            "created_at": u.created_at.strftime("%Y-%m-%d %H:%M") if u.created_at else "-"
        })
    return {
        "users": out,
        "pending_count": pending_count
    }


class UserApprovalPayload(BaseModel):
    user_id: int
    status: str  # 'APPROVED' atau 'REJECTED'
    username_admin: str = "admin"


@router.put("/api/admin/users/approval")
def set_user_approval(payload: UserApprovalPayload, request: Request, db: Session = Depends(get_db)):
    """Menyetujui atau menolak akun kasir baru."""
    target_user = db.query(User).filter(User.id == payload.user_id).first()
    if not target_user:
        raise HTTPException(status_code=404, detail="Pengguna tidak ditemukan")
    if target_user.role == "admin":
        raise HTTPException(status_code=400, detail="Akun Admin utama tidak dapat diubah statusnya")

    target_user.status_approval = payload.status.upper()
    client_ip, client_loc = detect_client_info(request)

    act_text = "MENYETUJUI" if payload.status.upper() == "APPROVED" else "MENOLAK"
    db.add(AuditActionLog(
        username=payload.username_admin,
        action_type="USER_APPROVAL",
        description=f"Admin {act_text} aktivasi akun kasir '{target_user.username}'",
        client_ip=client_ip,
        client_location=client_loc
    ))
    db.commit()
    return {"status": "SUCCESS", "message": f"Status akun {target_user.username} berhasil diubah menjadi {target_user.status_approval}"}


class UserPermissionPayload(BaseModel):
    user_id: int
    allowed_tabs: List[str]
    username_admin: str = "admin"


@router.put("/api/admin/users/permissions")
def update_user_permissions(payload: UserPermissionPayload, request: Request, db: Session = Depends(get_db)):
    """Mengubah hak akses tab mana saja yang boleh dilihat/digunakan pengguna."""
    target_user = db.query(User).filter(User.id == payload.user_id).first()
    if not target_user:
        raise HTTPException(status_code=404, detail="Pengguna tidak ditemukan")
    if target_user.role == "admin":
        raise HTTPException(status_code=400, detail="Hak akses akun Admin utama tidak dapat dibatasi")

    target_user.allowed_tabs = json.dumps(payload.allowed_tabs)
    client_ip, client_loc = detect_client_info(request)
    db.add(AuditActionLog(
        username=payload.username_admin,
        action_type="PERMISSION_UPDATE",
        description=f"Admin memperbarui izin akses akun '{target_user.username}': {len(payload.allowed_tabs)} fitur aktif",
        client_ip=client_ip,
        client_location=client_loc
    ))
    db.commit()
    return {"status": "SUCCESS", "message": f"Izin akses untuk {target_user.username} berhasil disimpan"}


# ================= DASHBOARD SUMMARY =================
@router.get("/api/dashboard-summary")
def get_dashboard_summary(db: Session = Depends(get_db)):
    debts = get_all_debts_for_table(db, include_settled=True)
    total_debt = get_total_receivable_balance(db)

    cfg = db.query(MLSetting).first()
    init_cash = float(getattr(cfg, "initial_cash_balance", 0.0) or 0.0) if cfg else 0.0
    init_tf = float(getattr(cfg, "initial_transfer_balance", 0.0) or 0.0) if cfg else 0.0
    init_debt = float(getattr(cfg, "initial_debt_balance", 0.0) or 0.0) if cfg else 0.0

    raw_dirs = sorted([d for d in Path("data/raw").glob("*") if d.is_dir()], reverse=True)

    total_kumulatif_cash_brankas = init_cash
    total_kumulatif_net_tf = init_tf
    latest_db_items = []
    unmatched_mutations = []
    available_dates = [d.name for d in raw_dirs]

    for date_folder in raw_dirs:
        vm_path = date_folder / "uang_masuk_brankas.json"
        if vm_path.exists():
            try:
                vm_data = json.loads(vm_path.read_text(encoding="utf-8"))
                val = float(vm_data.get("nominal") or vm_data.get("uang_masuk_brankas") or 0.0)
                total_kumulatif_cash_brankas += val
            except Exception:
                pass

        mb_path = date_folder / "mutasi_db_bank.json"
        if mb_path.exists():
            try:
                mb_data = json.loads(mb_path.read_text(encoding="utf-8"))
                records = mb_data.get("records", [])
                cr = sum(r.get("nominal", 0.0) for r in records if r.get("tipe") == "CR")
                db_exp = sum(r.get("nominal", 0.0) for r in records if r.get("tipe") == "DB")
                total_kumulatif_net_tf += (cr - db_exp)

                if not unmatched_mutations:
                    unmatched_mutations = [
                        {**r, "id": idx} for idx, r in enumerate(records)
                        if r.get("status_matching") == "UNMATCHED" and r.get("tipe") == "CR"
                    ]
                    latest_db_items = [r for r in records if r.get("tipe") == "DB"]
            except Exception:
                pass

    pending_users = db.query(User).filter(User.status_approval == "PENDING").count()

    return {
        "total_cash_fisik": total_kumulatif_cash_brankas,
        "cash_brankas": total_kumulatif_cash_brankas,
        "total_active_debt": total_debt + init_debt,
        "net_transfer": total_kumulatif_net_tf,
        "debts_table": debts,
        "latest_db_items": latest_db_items,
        "unmatched_mutations": unmatched_mutations,
        "available_dates": available_dates,
        "pending_registrations_count": pending_users,
        "initial_balances": {
            "cash": init_cash,
            "transfer": init_tf,
            "debt": init_debt
        },
        "ml_settings": {
            "cash_hard_limit": cfg.cash_hard_limit if cfg else 10000.0,
            "vault_emergency_limit": cfg.vault_emergency_limit if cfg else 500000.0,
            "model_max_age_days": cfg.model_max_age_days if cfg else 7
        }
    }


# ================= RUN AUDIT =================
@router.post("/api/run-audit")
async def run_audit(
    request: Request,
    pdf_file: UploadFile = File(...),
    csv_file: UploadFile = File(...),
    vault_in: float = Form(...),
    laci_items_json: str = Form(...),
    brankas_items_json: str = Form(...),
    cash_adjustment: float = Form(0.0),
    override_existing: bool = Form(False),
    username: str = Form(...),
    db: Session = Depends(get_db)
):
    try:
        pdf_bytes = await pdf_file.read()
        csv_bytes = await csv_file.read()

        pdf_data = parse_sales_report_pdf(io.BytesIO(pdf_bytes))
        audit_date = pdf_data.get("audit_date")
        if not audit_date:
            raise HTTPException(status_code=400, detail="Tanggal transaksi pada berkas PDF tidak dapat diidentifikasi.")

        target_archive_dir = Path("data/raw") / audit_date
        existing_marker = target_archive_dir / "uang_masuk_brankas.json"

        if existing_marker.exists() and not override_existing:
            raise HTTPException(
                status_code=409,
                detail=f"Laporan transaksi tanggal {audit_date} sudah pernah diaudit. Centang opsi timpa data jika ingin memproses ulang!"
            )

        bank_mutations = parse_bank_mutation_csv(io.BytesIO(csv_bytes))
        today_sales = pdf_data.get("records", [])

        laci_items = json.loads(laci_items_json) if laci_items_json else []
        brankas_items = json.loads(brankas_items_json) if brankas_items_json else []

        save_daily_cash_files(
            audit_date=audit_date,
            vault_in=vault_in,
            drawer_items=laci_items,
            vault_items=brankas_items,
            input_by=username
        )

        record_daily_credit_sales(db, audit_date, today_sales)
        db.flush()

        reconcile_res = reconcile_daily_bank_mutations(db, audit_date, bank_mutations, today_sales)
        db.flush()

        total_laci = sum(float(i.get("qty", 1.0)) * float(i.get("unit_price", 0.0)) for i in laci_items)
        total_brankas_out = sum(float(i.get("qty", 1.0)) * float(i.get("unit_price", 0.0)) for i in brankas_items)
        total_cash_fisik_harian = total_laci + vault_in + cash_adjustment
        total_cash_pdf = float(pdf_data.get("total_cash", 0.0))

        discrepancy = total_cash_fisik_harian - total_cash_pdf

        cfg = db.query(MLSetting).first() or MLSetting()
        audit_eval = evaluate_daily_audit_risks(
            current_discrepancy=discrepancy,
            vault_expense=total_brankas_out,
            unmatched_transfers=reconcile_res.get("unmatched_transfers_list", []),
            daily_transactions=today_sales,
            cash_hard_limit=cfg.cash_hard_limit,
            vault_emergency_limit=cfg.vault_emergency_limit,
            model_max_age_days=cfg.model_max_age_days
        )

        client_ip, client_loc = detect_client_info(request)
        db.add(AuditActionLog(
            username=username,
            action_type="AUDIT_RUN",
            description=f"Audit {audit_date}: Brankas Masuk=Rp {vault_in:,.0f}, Tunai PDF=Rp {total_cash_pdf:,.0f}, Selisih Kas=Rp {discrepancy:,.0f}",
            client_ip=client_ip,
            client_location=client_loc
        ))

        db.commit()

        return {
            "status": "SUCCESS",
            "audit_date": audit_date,
            "cash_collected": total_cash_fisik_harian,
            "system_cash_pdf": total_cash_pdf,
            "discrepancy": discrepancy,
            "ml_evaluation": audit_eval,
            "unmatched_list": reconcile_res.get("unmatched_transfers_list", [])
        }
    except HTTPException:
        db.rollback()
        raise
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Gagal memproses audit: {str(exc)}")


# ================= PERPUSTAKAAN DOKUMEN =================
@router.get("/api/documents")
def get_documents(db: Session = Depends(get_db)):
    docs = db.query(UploadedDocument).order_by(UploadedDocument.doc_date.desc(), UploadedDocument.id.desc()).all()
    return [
        {
            "id": d.id,
            "filename": d.filename,
            "file_type": d.file_type,
            "file_size_kb": d.file_size_kb,
            "extra_info": d.extra_info,
            "doc_date": d.doc_date,
            "upload_time": d.upload_time,
            "status": d.status
        }
        for d in docs
    ]


@router.post("/api/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    doc_date: str = Form(...),
    db: Session = Depends(get_db)
):
    ext = file.filename.split(".")[-1].upper() if "." in file.filename else "FILE"
    ftype = "PDF" if ext == "PDF" else ("CSV" if ext == "CSV" else ("IMG" if ext in ["JPG", "JPEG", "PNG"] else "OTHER"))

    safe_name = f"{doc_date}_{int(datetime.utcnow().timestamp())}_{file.filename}"
    saved_path = DOCS_DIR / safe_name
    content = await file.read()
    saved_path.write_bytes(content)

    size_kb = round(len(content) / 1024, 1)
    extra = "1 halaman" if ftype == "PDF" else ("File Data" if ftype == "CSV" else "Lampiran berkas")

    new_doc = UploadedDocument(
        filename=file.filename,
        file_path=str(saved_path),
        file_type=ftype,
        file_size_kb=size_kb,
        extra_info=extra,
        doc_date=doc_date,
        upload_time=datetime.now().strftime("%H:%M"),
        status="Belum Diproses"
    )
    db.add(new_doc)
    db.commit()
    return {"status": "SUCCESS", "message": "Dokumen berhasil disimpan ke perpustakaan"}


@router.get("/api/documents/{doc_id}/download")
def download_document(doc_id: int, db: Session = Depends(get_db)):
    doc = db.query(UploadedDocument).filter(UploadedDocument.id == doc_id).first()
    if not doc or not Path(doc.file_path).exists():
        raise HTTPException(status_code=404, detail="Berkas file tidak ditemukan di sistem")

    doc.status = "Terproses"
    db.commit()

    return FileResponse(path=doc.file_path, filename=doc.filename, media_type="application/octet-stream")


class StatusTogglePayload(BaseModel):
    status: str


@router.put("/api/documents/{doc_id}/status")
def toggle_document_status(doc_id: int, payload: StatusTogglePayload, db: Session = Depends(get_db)):
    doc = db.query(UploadedDocument).filter(UploadedDocument.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Dokumen tidak ditemukan")
    doc.status = payload.status
    db.commit()
    return {"status": "SUCCESS", "current_status": doc.status}


# ================= SET INITIAL BALANCES =================
class InitialBalancePayload(BaseModel):
    initial_cash: float = 0.0
    initial_transfer: float = 0.0
    initial_debt: float = 0.0
    username: str = "admin"


@router.post("/api/admin/set-initial-balances")
def set_initial_balances(payload: InitialBalancePayload, request: Request, db: Session = Depends(get_db)):
    cfg = db.query(MLSetting).first()
    if not cfg:
        cfg = MLSetting()
        db.add(cfg)

    cfg.initial_cash_balance = payload.initial_cash
    cfg.initial_transfer_balance = payload.initial_transfer
    cfg.initial_debt_balance = payload.initial_debt
    cfg.updated_by = payload.username
    cfg.updated_at = datetime.utcnow()

    client_ip, client_loc = detect_client_info(request)
    db.add(AuditActionLog(
        username=payload.username,
        action_type="BALANCE_ADJUSTMENT",
        description=f"Penyesuaian Saldo Awal: Cash=Rp {payload.initial_cash:,.0f}, TF=Rp {payload.initial_transfer:,.0f}, Utang=Rp {payload.initial_debt:,.0f}",
        client_ip=client_ip,
        client_location=client_loc
    ))
    db.commit()
    return {"status": "SUCCESS", "message": "Saldo awal berhasil disimpan dan diterapkan!"}


# ================= ML SETTINGS UPDATE =================
class MLSettingPayload(BaseModel):
    cash_hard_limit: float
    vault_emergency_limit: float
    model_max_age_days: int
    username: str = "admin"


@router.post("/api/admin/ml-settings")
def update_ml_settings(payload: MLSettingPayload, request: Request, db: Session = Depends(get_db)):
    cfg = db.query(MLSetting).first()
    if not cfg:
        cfg = MLSetting()
        db.add(cfg)

    cfg.cash_hard_limit = payload.cash_hard_limit
    cfg.vault_emergency_limit = payload.vault_emergency_limit
    cfg.model_max_age_days = payload.model_max_age_days
    cfg.updated_by = payload.username
    cfg.updated_at = datetime.utcnow()

    client_ip, client_loc = detect_client_info(request)
    db.add(AuditActionLog(
        username=payload.username,
        action_type="SETTING_UPDATE",
        description=f"Update Parameter ML: Limit Cash={payload.cash_hard_limit}, Limit Brankas={payload.vault_emergency_limit}, Age={payload.model_max_age_days} hari",
        client_ip=client_ip,
        client_location=client_loc
    ))
    db.commit()
    return {"status": "SUCCESS", "message": "Konfigurasi ML berhasil diperbarui"}


# ================= REKONSILIASI MANUAL =================
class BatchManualMatchPayload(BaseModel):
    audit_date: str
    selected_mutation_indices: List[int]
    target_receivable_id: Optional[int] = None
    target_customer_code: Optional[str] = None
    note: str
    username: str


@router.post("/api/batch-manual-match")
def batch_manual_match(payload: BatchManualMatchPayload, request: Request, db: Session = Depends(get_db)):
    folder = Path("data/raw") / payload.audit_date
    mb_file = folder / "mutasi_db_bank.json"
    if not mb_file.exists():
        raise HTTPException(status_code=404, detail="Berkas mutasi bank pada tanggal ini tidak ditemukan")

    data = json.loads(mb_file.read_text(encoding="utf-8"))
    records = data.get("records", [])

    matched_nominal_total = 0.0
    for idx in payload.selected_mutation_indices:
        if 0 <= idx < len(records):
            records[idx]["status_matching"] = "MATCHED_MANUAL"
            records[idx]["catatan_admin"] = payload.note
            records[idx]["matched_with"] = payload.target_customer_code or f"BON-{payload.target_receivable_id}"
            matched_nominal_total += float(records[idx].get("nominal", 0.0))

    if payload.target_receivable_id:
        settle_receivable(db, payload.target_receivable_id, matched_nominal_total, payload.audit_date)

    data["records"] = records
    mb_file.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    client_ip, client_loc = detect_client_info(request)
    db.add(AuditActionLog(
        username=payload.username,
        action_type="MANUAL_MATCH",
        description=f"Match Manual: {len(payload.selected_mutation_indices)} mutasi CR dipasangkan ke '{payload.target_customer_code or payload.target_receivable_id}' (Total Rp {matched_nominal_total:,.0f})",
        client_ip=client_ip,
        client_location=client_loc
    ))
    db.commit()
    return {"status": "SUCCESS", "matched_total": matched_nominal_total}


# ================= MASTER DATA PELANGGAN =================
class CustomerPayload(BaseModel):
    customer_code: str
    name: str
    address: str
    phone: str = ""


@router.get("/api/admin/customers")
def get_customers(db: Session = Depends(get_db)):
    custs = db.query(Customer).order_by(Customer.customer_code.asc()).all()
    return [
        {
            "id": c.id,
            "code": c.customer_code,
            "name": c.name,
            "address": c.address_raw or "-",
            "phone": c.phone or "-"
        }
        for c in custs
    ]


@router.post("/api/admin/customers")
def save_customer(payload: CustomerPayload, db: Session = Depends(get_db)):
    code_u = payload.customer_code.strip().upper()
    cust = db.query(Customer).filter(Customer.customer_code == code_u).first()
    if cust:
        cust.name = payload.name.strip()
        cust.address_raw = payload.address.strip()
        cust.address_clean = payload.address.strip()
        cust.phone = payload.phone.strip()
    else:
        db.add(Customer(
            customer_code=code_u,
            name=payload.name.strip(),
            address_raw=payload.address.strip(),
            address_clean=payload.address.strip(),
            phone=payload.phone.strip()
        ))
    db.commit()
    return {"status": "SUCCESS"}


# ================= AUDIT LOGS & BERKAS ARSIP =================
@router.get("/api/admin/audit-logs")
def get_audit_logs(db: Session = Depends(get_db)):
    logs = db.query(AuditActionLog).order_by(AuditActionLog.timestamp.desc()).limit(200).all()
    return [
        {
            "id": l.id,
            "waktu": l.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "username": l.username,
            "aksi": l.action_type,
            "deskripsi": l.description,
            "ip": l.client_ip or "-",
            "lokasi": getattr(l, "client_location", "Lokal / Internal") or "Lokal"
        }
        for l in logs
    ]


@router.get("/api/folder-files/{date_str}")
def get_folder_files(date_str: str):
    folder = Path("data/raw") / date_str
    if not folder.exists():
        raise HTTPException(status_code=404, detail="Folder arsip tidak ditemukan")

    files_result = {}
    for f in folder.glob("*.json"):
        try:
            files_result[f.name] = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            files_result[f.name] = {}
    return files_result