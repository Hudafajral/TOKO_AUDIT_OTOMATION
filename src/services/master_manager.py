from typing import Dict, List, Optional
from sqlalchemy.orm import Session
from src.models.audit_models import Customer, CustomerAlias, ClusterAbbr
from src.services.normalizer import normalize_address

# ==========================================
# MANAJEMEN MASTER PELANGGAN & ALIAS
# ==========================================

def get_or_create_customer(
    db: Session,
    customer_code: str,
    address_raw: str,
    phone: Optional[str] = None
) -> Customer:
    """
    Mengambil data pelanggan berdasarkan kode akun.
    Jika belum terdaftar di database, otomatis dibuatkan baris baru.
    """
    clean_code = customer_code.upper().strip()
    customer = db.query(Customer).filter(Customer.customer_code == clean_code).first()
    
    if not customer:
        clean_addr = normalize_address(address_raw)
        customer = Customer(
            customer_code=clean_code,
            address_raw=address_raw.strip(),
            address_clean=clean_addr,
            phone=phone
        )
        db.add(customer)
        db.commit()
        db.refresh(customer)
        
    return customer

def add_customer_alias(
    db: Session,
    customer_id: int,
    alias_name: str
) -> CustomerAlias:
    """
    Mendaftarkan variasi nama rekening baru untuk seorang pelanggan.
    """
    clean_alias = alias_name.upper().strip()
    existing = db.query(CustomerAlias).filter(
        CustomerAlias.customer_id == customer_id,
        CustomerAlias.alias_name == clean_alias
    ).first()
    
    if existing:
        return existing
        
    new_alias = CustomerAlias(
        customer_id=customer_id,
        alias_name=clean_alias
    )
    db.add(new_alias)
    db.commit()
    db.refresh(new_alias)
    return new_alias

def get_all_customers(db: Session) -> List[Customer]:
    """Mengambil seluruh daftar master pelanggan Toko Rina."""
    return db.query(Customer).order_by(Customer.customer_code.asc()).all()


# ==========================================
# MANAJEMEN KAMUS SINGKATAN CLUSTER
# ==========================================

def load_cluster_dictionary(db: Session) -> Dict[str, str]:
    """
    Mengambil seluruh pemetaan singkatan cluster dari database
    dalam bentuk dictionary python, contoh: {'PS': 'PASADENA', 'MLD': 'MALIBU'}.
    """
    records = db.query(ClusterAbbr).all()
    return {item.abbr_code.upper(): item.full_name.upper() for item in records}

def set_cluster_abbreviation(
    db: Session,
    abbr_code: str,
    full_name: str
) -> ClusterAbbr:
    """
    Menambah atau memperbarui kepanjangan singkatan cluster.
    """
    clean_code = abbr_code.upper().strip()
    clean_full = full_name.upper().strip()
    
    entry = db.query(ClusterAbbr).filter(ClusterAbbr.abbr_code == clean_code).first()
    if entry:
        entry.full_name = clean_full
    else:
        entry = ClusterAbbr(abbr_code=clean_code, full_name=clean_full)
        db.add(entry)
        
    db.commit()
    db.refresh(entry)
    return entry