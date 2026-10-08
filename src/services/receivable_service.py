"""
src/services/receivable_service.py
Layanan manajemen piutang bon konsumen Toko Rina.
Hanya mendaftarkan pelanggan ke Master Pelanggan jika memiliki transaksi Kredit (Bon).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session

from src.models.audit_models import Customer, Receivable


def record_daily_credit_sales(
    db: Session,
    audit_date: str,
    sales_records: List[Dict[str, Any]]
) -> List[Receivable]:
    """
    Hanya mencatat transaksi Kredit (Bon) ke Buku Piutang dan Master Pelanggan.
    Pelanggan umum/tunai/debit murni tidak dimasukkan ke Master Pelanggan.
    """
    created_receivables: List[Receivable] = []

    for s in sales_records:
        kredit = float(s.get("kredit") or s.get("credit") or 0.0)
        if kredit <= 0:
            continue

        raw_code = str(s.get("customer_code") or "").strip()
        if not raw_code or "TOTAL" in raw_code.upper():
            continue

        clean_code = " ".join(raw_code.upper().split())

        # Daftarkan ke Master Pelanggan hanya yang memiliki transaksi Bon/Kredit
        customer = db.query(Customer).filter(Customer.customer_code == clean_code).first()
        if not customer:
            customer = Customer(
                customer_code=clean_code,
                name=clean_code,
                address_raw=clean_code,
                address_clean=clean_code
            )
            db.add(customer)
            db.flush()

        inv_num = f"BON-{audit_date}-{clean_code.replace(' ', '')}"

        existing = db.query(Receivable).filter(
            Receivable.customer_id == customer.id,
            Receivable.sale_date == audit_date
        ).first()

        if not existing:
            new_rec = Receivable(
                customer_id=customer.id,
                sale_date=audit_date,
                sale_invoice=inv_num,
                initial_amount=kredit,
                remaining_amount=kredit,
                status="BELUM",
                paid_date=None
            )
            db.add(new_rec)
            created_receivables.append(new_rec)

    db.flush()
    return created_receivables


def settle_receivable(
    db: Session,
    receivable_id: int,
    payment_amount: float,
    settle_date: str
) -> Optional[Receivable]:
    """Melunasi piutang bon di database dan mengubah status menjadi LUNAS."""
    rec = db.query(Receivable).filter(Receivable.id == receivable_id).first()
    if not rec:
        return None

    current_rem = float(rec.remaining_amount or 0.0)
    pay = float(payment_amount or 0.0)

    new_rem = max(0.0, current_rem - pay)
    rec.remaining_amount = new_rem

    if new_rem <= 1.0:
        rec.remaining_amount = 0.0
        rec.status = "LUNAS"
        rec.paid_date = settle_date
    else:
        rec.status = "SEBAGIAN"

    db.add(rec)
    db.flush()
    return rec


# Alias fungsi agar kompatibel dengan pemanggilan fungsi lama maupun baru
settle_receivable_direct = settle_receivable


def get_all_debts_for_table(db: Session, include_settled: bool = True) -> List[Dict[str, Any]]:
    query = db.query(Receivable)
    if not include_settled:
        query = query.filter(Receivable.status != "LUNAS")

    records = query.order_by(Receivable.sale_date.desc(), Receivable.id.desc()).all()
    results = []

    for r in records:
        cust_label = r.customer.customer_code if r.customer else f"ID-{r.id}"
        results.append({
            "ID": r.id,
            "Tanggal Nota": r.sale_date or "-",
            "Kode / Nama Pelanggan": cust_label,
            "No. Faktur": r.sale_invoice or f"BON-{r.id}",
            "Total Bon Asli": float(r.initial_amount or 0.0),
            "Sisa Utang": float(r.remaining_amount or 0.0),
            "Status": r.status or "BELUM",
            "Tanggal Lunas": r.paid_date or "-"
        })

    return results


def get_total_receivable_balance(db: Session) -> float:
    records = db.query(Receivable).filter(Receivable.status != "LUNAS").all()
    return sum(float(r.remaining_amount or 0.0) for r in records)