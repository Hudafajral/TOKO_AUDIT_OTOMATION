"""
src/routers/audit_router.py
Router FastAPI untuk orkestrasi audit harian dan retrain model deteksi anomali.
"""
from typing import Any, Dict, List
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from src.schemas.audit_schema import DailyAuditRequest, DailyAuditResponse
from src.services.anomaly_ml_service import (
    evaluate_daily_audit_risks,
    retrain_models_if_needed,
    benchmark_ml_vs_baselines,
)

router = APIRouter(
    prefix="/audit",
    tags=["Audit & Anomaly Detection"]
)


class RetrainRequest(BaseModel):
    historical_discrepancies: List[float] = Field(
        ...,
        description="Riwayat selisih kas fisik (minimal 90 hari)",
        example=[0.0] * 90
    )
    historical_transactions: List[Dict[str, Any]] = Field(
        ...,
        description="Riwayat transaksi penjualan (minimal 500 baris)"
    )
    force: bool = Field(
        False,
        description="Paksa latih ulang meskipun model belum kedaluwarsa"
    )


@router.post(
    "/daily-evaluation",
    response_model=DailyAuditResponse,
    status_code=status.HTTP_200_OK,
    summary="Evaluasi Risiko Audit Harian (Hibrida: Aturan + ML)"
)
def run_daily_audit_evaluation(payload: DailyAuditRequest):
    """
    Menjalankan audit rekonsiliasi harian toko:
    - Mengecek aturan operasional keras (kasir, pengeluaran brankas, limit kas fisik).
    - Menjalankan inferensi model ML (One-Class SVM & Isolation Forest).
    - Mengembalikan status: HEALTHY, HEALTHY_PARTIAL, REVIEW_SUGGESTED, atau ATTENTION_REQUIRED.
    """
    try:
        # Konversi objek TransactionItem Pydantic ke format dictionary untuk service ML
        tx_dicts = (
            [tx.model_dump() for tx in payload.daily_transactions]
            if payload.daily_transactions
            else []
        )

        result = evaluate_daily_audit_risks(
            current_discrepancy=payload.current_discrepancy,
            vault_expense=payload.vault_expense,
            unmatched_transfers=payload.unmatched_transfers,
            daily_transactions=tx_dicts,
        )
        return result
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Validasi payload transaksi gagal: {str(val_err)}"
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Terjadi kesalahan internal pada layanan audit: {str(exc)}"
        )


@router.post(
    "/retrain",
    status_code=status.HTTP_200_OK,
    summary="Jalankan Pelatihan Ulang Model ML Audit"
)
def trigger_model_retrain(payload: RetrainRequest):
    """
    Melatih ulang bundle model One-Class SVM dan Isolation Forest
    jika data telah mencukupi atau masa berlaku model (>7 hari) telah terlewati.
    """
    try:
        train_result = retrain_models_if_needed(
            historical_discrepancies=payload.historical_discrepancies,
            historical_transactions=payload.historical_transactions,
            force=payload.force,
        )
        return {
            "message": "Pemeriksaan dan pelatihan model berhasil dijalankan.",
            "details": train_result
        }
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Gagal melakukan retrain model: {str(exc)}"
        )


@router.post(
    "/benchmark",
    status_code=status.HTTP_200_OK,
    summary="Uji Benchmark Empiris: ML vs Baseline Statistik (IQR & Heuristik)"
)
def run_benchmark(transactions: List[Dict[str, Any]]):
    """
    Menjalankan pengujian multi-seed pada 4 skenario anomali sintetis
    untuk membuktikan nilai tambah ML terhadap aturan heuristik toko.
    """
    try:
        benchmark_result = benchmark_ml_vs_baselines(transactions)
        return benchmark_result
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Gagal menjalankan benchmark: {str(exc)}"
        )