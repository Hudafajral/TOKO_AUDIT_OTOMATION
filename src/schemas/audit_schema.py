"""
src/schemas/audit_schema.py
Skema Pydantic untuk request dan response audit harian.
"""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class TransactionItem(BaseModel):
    date: str = Field(..., example="2026-09-30")
    customer_code: str = Field(..., example="CUST_A")
    total: float = Field(..., example=150000.0)
    cash: float = Field(0.0, example=50000.0)
    transfer: float = Field(0.0, example=100000.0)
    credit: float = Field(0.0, example=0.0)


class DailyAuditRequest(BaseModel):
    current_discrepancy: float = Field(
        ..., 
        description="Selisih kas fisik vs pencatatan sistem (bisa positif/negatif)",
        example=5000.0
    )
    vault_expense: float = Field(
        0.0, 
        description="Pengeluaran darurat langsung dari brankas",
        example=0.0
    )
    unmatched_transfers: Optional[List[Dict[str, Any]]] = Field(
        default_factory=list,
        description="Daftar mutasi transfer yang belum cocok dengan nota"
    )
    daily_transactions: Optional[List[TransactionItem]] = Field(
        default_factory=list,
        description="Daftar transaksi penjualan hari ini"
    )


class DailyAuditResponse(BaseModel):
    status: str = Field(..., example="HEALTHY")
    component_coverage: Dict[str, str]
    risk_flags: List[str]
    review_flags: List[str]
    cash_audit_detail: Dict[str, Any]
    transaction_audit_detail: Dict[str, Any]