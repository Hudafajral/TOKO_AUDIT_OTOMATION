from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from src.models.audit_models import BankLedgerDB

def record_bank_debit(
    db: Session,
    trx_date: str,
    amount_out: float,
    recipient_desc: str,
    current_balance: float
) -> BankLedgerDB:
    """
    Mencatat pengeluaran mutasi bank (DB) dan menghitung saldo berjalan baru.
    Running Balance Baru = Saldo Sebelumnya - Pengeluaran (amount_out)
    """
    new_balance = current_balance - amount_out
    
    ledger_entry = BankLedgerDB(
        trx_date=trx_date,
        amount_out=amount_out,
        recipient_desc=recipient_desc,
        running_balance=new_balance
    )
    db.add(ledger_entry)
    db.commit()
    db.refresh(ledger_entry)
    return ledger_entry

def process_bank_debit_mutations(
    db: Session,
    bank_transactions: List[Dict[str, Any]],
    initial_balance: float
) -> List[BankLedgerDB]:
    """
    Memfilter seluruh transaksi bertipe 'DB' (uang keluar) dari mutasi bank,
    kemudian menyimpannya secara berurutan sambil memperbarui running balance.
    """
    saved_entries = []
    current_balance = initial_balance
    
    for trx in bank_transactions:
        if trx.get("type") == "DB":
            amount = trx.get("amount", 0.0)
            date = trx.get("trx_date", "")
            desc = trx.get("description", "")
            
            entry = record_bank_debit(
                db=db,
                trx_date=date,
                amount_out=amount,
                recipient_desc=desc,
                current_balance=current_balance
            )
            current_balance = entry.running_balance
            saved_entries.append(entry)
            
    return saved_entries

def get_latest_bank_balance(db: Session, fallback_balance: float = 0.0) -> float:
    """
    Mengambil saldo bank terkini untuk ditampilkan pada Card Debit/Bank di dashboard utama.
    Jika belum ada mutasi DB yang tercatat, mengembalikan nilai fallback_balance.
    """
    last_entry = db.query(BankLedgerDB).order_by(BankLedgerDB.id.desc()).first()
    if last_entry:
        return last_entry.running_balance
    return fallback_balance