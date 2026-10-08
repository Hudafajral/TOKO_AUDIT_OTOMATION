"""
src/services/transfer_service.py
Layanan rekonsiliasi transfer bank Toko Rina.
Urutan Eksekusi:
1. Cek Penjualan Hari Ini di PDF (K.Debit & Bon Kredit Hari Ini).
2. Jika tidak ada, cek Tabel Piutang database.
3. Verifikasi nama & kode blok (D 20 vs H 36) agar tidak tertukar.
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session

from src.models.audit_models import Customer, Receivable
from src.services.receivable_service import settle_receivable_direct


def clean_text_normalize(text_val: Any) -> str:
    """Normalisasi huruf Unicode ke alfabet Latin standar."""
    if text_val is None:
        return ""
    s = str(text_val)
    norm = unicodedata.normalize("NFKD", s)
    char_map = {
        "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I",
        "Κ": "K", "Μ": "M", "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T",
        "Υ": "Y", "Χ": "X", "а": "A", "е": "E", "о": "O", "р": "P",
        "с": "C", "у": "Y", "х": "X"
    }
    for k, v in char_map.items():
        norm = norm.replace(k, v)
    cleaned = re.sub(r"[^A-Za-z0-9\s]", " ", norm).upper()
    return " ".join(cleaned.split())


def extract_block_code(text_val: str) -> Optional[str]:
    """Ekstraksi kode blok (contoh: 'PASADENA D 20' -> 'D20', 'H 36' -> 'H36')."""
    clean = clean_text_normalize(text_val)
    m = re.search(r"\b([A-Z])\s*(\d{1,3})\b", clean)
    if m:
        return f"{m.group(1)}{m.group(2)}"
    return None


def is_strictly_matched(cust_name: str, desc_text: str) -> bool:
    """
    Pencocokan presisi:
    Wajib mencocokkan kode blok secara identik (D20 != H36).
    """
    cust_clean = clean_text_normalize(cust_name)
    desc_clean = clean_text_normalize(desc_text)
    desc_compact = desc_clean.replace(" ", "")

    c_block = extract_block_code(cust_clean)
    if c_block:
        d_block = extract_block_code(desc_clean)
        if d_block and d_block != c_block:
            return False
        block_spaced = f"{c_block[0]} {c_block[1:]}"
        return (c_block in desc_compact) or (block_spaced in desc_clean)

    cust_compact = cust_clean.replace(" ", "")
    return (cust_compact in desc_compact) if len(cust_compact) >= 4 else False


def reconcile_daily_bank_mutations(
    db: Session,
    audit_date: str,
    mutations: List[Dict[str, Any]],
    today_sales: List[Dict[str, Any]]
) -> Dict[str, Any]:
    folder_path = Path("data/raw") / audit_date
    folder_path.mkdir(parents=True, exist_ok=True)
    mb_file = folder_path / "mutasi_db_bank.json"

    db.flush()

    # 1. Siapkan K.Debit Penjualan Hari Ini dari PDF
    today_debit_sales = []
    # 2. Siapkan Bon/Kredit Penjualan Hari Ini dari PDF
    today_credit_sales = []

    for s in today_sales:
        c_code = str(s.get("customer_code") or "").strip()
        k_deb = float(s.get("k_debit") or s.get("debit") or 0.0)
        k_kred = float(s.get("kredit") or s.get("credit") or 0.0)

        if k_deb > 0 and c_code:
            today_debit_sales.append({
                "customer_code": c_code,
                "nominal": k_deb,
                "is_matched": False
            })
        if k_kred > 0 and c_code:
            today_credit_sales.append({
                "customer_code": c_code,
                "nominal": k_kred,
                "is_matched": False
            })

    # 3. Siapkan Seluruh Piutang Aktif di Database
    active_recs = db.query(Receivable).filter(Receivable.status != "LUNAS").all()
    debt_items = []
    for r in active_recs:
        c_code = r.customer.customer_code if r.customer else (getattr(r, "customer_code_raw", "") or "")
        debt_items.append({
            "id": r.id,
            "sale_date": r.sale_date,
            "customer_code": c_code,
            "sale_invoice": r.sale_invoice or f"BON-{r.id}",
            "remaining_amount": float(r.remaining_amount or 0.0),
            "is_settled": False
        })

    reconciled_records = []
    unmatched_transfers = []

    # 4. Rekonsiliasi Setiap Baris Mutasi Bank
    for idx, m in enumerate(mutations):
        m_date = str(m.get("date") or audit_date)
        m_desc = str(m.get("description") or "")
        m_type = str(m.get("type") or "CR").upper()
        m_amount = float(m.get("amount") or 0.0)

        record_entry = {
            "id": idx,
            "tanggal": m_date,
            "keterangan": m_desc,
            "tipe": m_type,
            "nominal": m_amount,
            "status_matching": "UNMATCHED",
            "matched_with": None,
            "catatan_admin": ""
        }

        if m_type == "DB":
            record_entry["status_matching"] = "EXPENSE_DB"
            record_entry["matched_with"] = "PENGELUARAN_OPERASIONAL"
            reconciled_records.append(record_entry)
            continue

        # =========================================================
        # TAHAP 1: CEK PENJUALAN HARI INI (BON / K.DEBIT PDF)
        # =========================================================
        matched_today = False

        # 1A. Cek Pembayaran Bon Hari Ini (misal PASADENA D 20)
        for cs in today_credit_sales:
            if cs["is_matched"]:
                continue
            if abs(cs["nominal"] - m_amount) < 1.0:
                if is_strictly_matched(cs["customer_code"], m_desc):
                    cs["is_matched"] = True
                    matched_today = True
                    record_entry["status_matching"] = "MATCHED_DEBT_SETTLED"
                    record_entry["matched_with"] = f"{cs['customer_code']} (Bon PDF Hari Ini)"
                    record_entry["catatan_admin"] = f"Pelunasan Bon Hari Ini ({cs['customer_code']})"

                    # Update status piutang tanggal hari ini menjadi LUNAS
                    for d in debt_items:
                        if not d["is_settled"] and d["sale_date"] == audit_date:
                            if is_strictly_matched(d["customer_code"], cs["customer_code"]):
                                d["is_settled"] = True
                                settle_receivable_direct(db, d["id"], m_amount, audit_date)
                                break
                    break

        if matched_today:
            reconciled_records.append(record_entry)
            continue

        # 1B. Cek Transaksi K.Debit Hari Ini
        for ds in today_debit_sales:
            if ds["is_matched"]:
                continue
            if abs(ds["nominal"] - m_amount) < 1.0:
                if is_strictly_matched(ds["customer_code"], m_desc):
                    ds["is_matched"] = True
                    matched_today = True
                    record_entry["status_matching"] = "MATCHED_SALES_DEBIT"
                    record_entry["matched_with"] = ds["customer_code"]
                    record_entry["catatan_admin"] = f"K.Debit POS ({ds['customer_code']})"
                    break

        if matched_today:
            reconciled_records.append(record_entry)
            continue

        # =========================================================
        # TAHAP 2: CEK TABEL PIUTANG DATABASE LAMA
        # =========================================================
        matched_old_debt = False
        for d in debt_items:
            if d["is_settled"]:
                continue
            if abs(d["remaining_amount"] - m_amount) < 1.0 or m_amount <= d["remaining_amount"]:
                if is_strictly_matched(d["customer_code"], m_desc):
                    d["is_settled"] = True
                    matched_old_debt = True
                    record_entry["status_matching"] = "MATCHED_DEBT_SETTLED"
                    record_entry["matched_with"] = f"{d['customer_code']} ({d['sale_invoice']})"
                    record_entry["catatan_admin"] = f"Pelunasan Piutang Bon ({d['sale_invoice']})"
                    settle_receivable_direct(db, d["id"], m_amount, audit_date)
                    break

        if matched_old_debt:
            reconciled_records.append(record_entry)
            continue

        unmatched_transfers.append({
            "index": idx,
            "date": m_date,
            "description": m_desc,
            "amount": m_amount
        })
        reconciled_records.append(record_entry)

    output_data = {
        "audit_date": audit_date,
        "total_records": len(reconciled_records),
        "total_unmatched_cr": len(unmatched_transfers),
        "records": reconciled_records
    }
    mb_file.write_text(json.dumps(output_data, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "status": "SUCCESS",
        "total_mutations": len(mutations),
        "unmatched_transfers_list": unmatched_transfers,
        "reconciled_records": reconciled_records
    }