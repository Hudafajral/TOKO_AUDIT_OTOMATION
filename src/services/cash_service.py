"""
src/services/cash_service.py
Layanan rekonsiliasi kas fisik berbasis rincian item belanja laci dan brankas.
"""
from typing import Dict, Any, List

def calculate_itemized_expenses(items: List[Dict[str, Any]]) -> float:
    """Menghitung total nominal dari daftar item (qty * unit_price)."""
    total = 0.0
    for item in items:
        qty = float(item.get("qty", 1.0))
        price = float(item.get("unit_price", 0.0))
        total += qty * price
    return total

def calculate_cash_reconciliation(
    drawer_items: List[Dict[str, Any]],
    vault_cash_in: float,
    system_cash_sales: float,
    vault_items: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Rumus:
    1. Total Penjualan Fisik = Uang Keluar Laci (belanja harian) + Uang Masuk Brankas (setoran bersih).
    2. Selisih (Discrepancy) = Total Penjualan Fisik - Total Tunai Nota PDF.
    3. Pengeluaran Brankas = Total item belanja barang/kulakan jika uang laci kurang.
    """
    drawer_cash_out = calculate_itemized_expenses(drawer_items)
    vault_cash_out = calculate_itemized_expenses(vault_items)
    
    total_sales_cash_collected = drawer_cash_out + vault_cash_in
    discrepancy = total_sales_cash_collected - system_cash_sales

    return {
        "drawer_cash_out": drawer_cash_out,
        "vault_cash_in": vault_cash_in,
        "vault_cash_out": vault_cash_out,
        "total_sales_cash_collected": total_sales_cash_collected,
        "system_cash_sales": system_cash_sales,
        "discrepancy": discrepancy
    }