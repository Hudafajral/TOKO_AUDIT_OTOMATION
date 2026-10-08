"""
tests/test_anomaly_ml_service.py
Jalankan: python -m pytest -q tests/test_anomaly_ml_service.py
"""
import numpy as np
import pytest

import src.services.anomaly_ml_service as m

@pytest.fixture()
def tmp_models(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "CASH_BUNDLE_PATH", tmp_path / "cash.joblib")
    monkeypatch.setattr(m, "TX_BUNDLE_PATH", tmp_path / "tx.joblib")
    m._MODEL_CACHE.clear()
    return tmp_path

@pytest.fixture(scope="module")
def txs():
    return m._generate_demo_transactions()

def _odd(txs, total=25_000_000.0, credit=25_000_000.0, customer="NEW_CUST_99"):
    return {
        "date": "2026-09-15",
        "customer_code": customer,
        "total": total,
        "cash": 0.0,
        "transfer": 0.0,
        "credit": credit
    }

def test_fallback_tanpa_model_menangkap_utang_ekstrem(tmp_models, txs):
    r = m.evaluate_daily_audit_risks(0.0, 0, [], txs[-20:] + [_odd(txs)])
    assert r["status"] == "ATTENTION_REQUIRED"
    assert r["component_coverage"]["transaction_audit"] == "FALLBACK_HEURISTIC"

def test_aturan_dan_ml_aktif_setelah_model_dilatih(tmp_models, txs):
    cash = list(np.random.default_rng(1).normal(0, 3000, 120).round(-2))
    m.retrain_models_if_needed(cash, txs)
    r = m.evaluate_daily_audit_risks(0.0, 0, [], txs[-20:] + [_odd(txs)])
    assert r["component_coverage"]["transaction_audit"] == "ML_MODEL"
    assert r["status"] == "ATTENTION_REQUIRED"
    assert r["transaction_audit_detail"]["high_severity_count"] >= 1

def test_batas_keras_kas_selalu_risiko(tmp_models, txs):
    cash = list(np.random.default_rng(1).normal(0, 3000, 120).round(-2))
    m.retrain_models_if_needed(cash, txs)
    r = m.evaluate_daily_audit_risks(25_000.0, 0, [], txs[-20:])
    assert any("batas keras" in f for f in r["risk_flags"])

def test_kas_dominan_nol_memakai_aturan_empiris(tmp_models):
    hist = [0.0] * 95 + [500.0, -1000.0, 2000.0, 1500.0, -800.0] * 2
    assert m.train_cash_discrepancy_model(hist)["metadata"]["is_degenerate"] is True
    assert m.infer_cash_discrepancy_ml(0.0)["status"] == "EMPIRICAL_RULE"
    assert m.infer_cash_discrepancy_ml(0.0)["is_anomaly"] is False

def test_payload_tidak_valid_ditolak():
    ok = {"date": "2026-09-01", "customer_code": "CUST_A", "total": 10000.0}
    with pytest.raises(ValueError):
        m.validate_transaction_payload([dict(ok, total=-500.0)])
    with pytest.raises(ValueError):
        m.validate_transaction_payload([{k: v for k, v in ok.items() if k != "customer_code"}])
    with pytest.raises(ValueError):
        m.validate_transaction_payload([dict(ok, date="bukan-tanggal")])