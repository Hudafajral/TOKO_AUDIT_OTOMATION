from typing import Optional, Tuple, Dict, Any, List
from sqlalchemy.orm import Session
from rapidfuzz import fuzz, process
from src.models.audit_models import Customer, CustomerAlias
from src.config import settings

def find_customer_match(
    db: Session,
    raw_sender_name: str
) -> Tuple[Optional[Customer], float, str]:
    """
    Mencari kecocokan pelanggan dari nama pengirim mutasi bank:
    1. Cek Exact Match pada Customer Alias.
    2. Cek Fuzzy Match pada Customer Alias menggunakan RapidFuzz.
    3. Cek Fuzzy Match pada Alamat Pelanggan.
    
    Mengembalikan: (Customer object, skor kemiripan 0-100, alasan pencocokan)
    """
    if not raw_sender_name:
        return None, 0.0, "Nama pengirim kosong"

    clean_sender = raw_sender_name.upper().strip()

    # 1. Pengecekan Exact Match pada Alias
    exact_alias = db.query(CustomerAlias).filter(CustomerAlias.alias_name == clean_sender).first()
    if exact_alias:
        return exact_alias.customer, 100.0, "EXACT_ALIAS_MATCH"

    # Ambil seluruh alias dari database untuk Fuzzy Matching
    aliases = db.query(CustomerAlias).all()
    if aliases:
        alias_dict = {a.alias_name: a.customer for a in aliases}
        best_match = process.extractOne(
            clean_sender,
            list(alias_dict.keys()),
            scorer=fuzz.token_sort_ratio
        )
        if best_match:
            matched_name, score, _ = best_match
            if score >= settings.FUZZY_MATCH_THRESHOLD:
                return alias_dict[matched_name], float(score), f"FUZZY_ALIAS_MATCH ({matched_name})"

    # 2. Pengecekan Fuzzy Match pada Alamat Pelanggan (jika mutasi mencantumkan blok/cluster)
    customers = db.query(Customer).all()
    if customers:
        addr_dict = {c.address_clean: c for c in customers if c.address_clean}
        best_addr = process.extractOne(
            clean_sender,
            list(addr_dict.keys()),
            scorer=fuzz.partial_ratio
        )
        if best_addr:
            matched_addr, score, _ = best_addr
            if score >= settings.FUZZY_MATCH_THRESHOLD:
                return addr_dict[matched_addr], float(score), f"FUZZY_ADDRESS_MATCH ({matched_addr})"

    return None, 0.0, "NO_MATCH_FOUND"

def register_new_alias(db: Session, customer_id: int, new_alias_name: str) -> CustomerAlias:
    """
    Menyimpan alias baru hasil konfirmasi review agar otomatis dikenali di transaksi berikutnya.
    """
    clean_alias = new_alias_name.upper().strip()
    existing = db.query(CustomerAlias).filter(
        CustomerAlias.customer_id == customer_id,
        CustomerAlias.alias_name == clean_alias
    ).first()

    if existing:
        return existing

    alias_record = CustomerAlias(customer_id=customer_id, alias_name=clean_alias)
    db.add(alias_record)
    db.commit()
    db.refresh(alias_record)
    return alias_record