"""
src/services/anomaly_ml_service.py
==================================
Layanan audit hibrida (aturan bisnis + machine learning) untuk Toko Rina.
Prinsip: Aturan operasional adalah fondasi utama; ML bertindak sebagai pelengkap.
"""
from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.svm import OneClassSVM

logger = logging.getLogger(__name__)

# ============================================================================
# KONFIGURASI SENTRAL TOKO RINA
# ============================================================================

@dataclass(frozen=True)
class AuditMLConfig:
    MIN_CASH_DAYS: int = 90
    MIN_TX_SAMPLES: int = 500
    MIN_BENCH_SAMPLES: int = 200
    MODEL_MAX_AGE_DAYS: int = 7

    CASH_HARD_LIMIT: float = 10_000.0          # Selisih kas > Rp 10.000 selalu RISIKO
    EMERGENCY_VAULT_LIMIT: float = 500_000.0   # Pengeluaran brankas darurat > Rp 500.000
    MAX_UNMATCHED_TRANSFERS: int = 3
    Z_SCORE_THRESHOLD: float = 2.5
    DEGENERATE_ZERO_RATIO: float = 0.80

    TX_FLAG_PERCENTILE: float = 1.0
    EXTREME_SPREAD_MULT: float = 2.0
    USE_ONEHOT_CATEGORICAL: bool = True
    MIN_REVIEW_FLAGS: int = 3
    REVIEW_FLAG_RATIO: float = 0.03

    BENCH_SAMPLES_PER_TYPE: int = 25
    BENCH_ASSUMED_PREVALENCE: float = 0.01
    BENCH_MARGIN: float = 0.05

CONFIG = AuditMLConfig()

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = Path(os.environ.get("AUDIT_MODEL_DIR", BASE_DIR / "data" / "models"))
CASH_BUNDLE_PATH = MODEL_DIR / "cash_bundle.joblib"
TX_BUNDLE_PATH = MODEL_DIR / "tx_isoforest_bundle.joblib"

REQUIRED_TX_FIELDS = ["date", "customer_code", "total"]
COMBO_COLS = ["customer_code", "payment_profile", "day_type"]
ANOMALY_TYPES = ["a_extreme_total", "b_unusual_credit_ratio", "c_novel_combo", "d_rare_profile"]

_MODEL_CACHE: Dict[str, Any] = {}


# ============================================================================
# 1. BUNDLE I/O ATOMIK & CACHE MULTI-WORKER
# ============================================================================

def save_bundle_atomic(bundle_data: Dict[str, Any], target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_name: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(dir=target_path.parent, delete=False) as tf:
            temp_name = tf.name
            joblib.dump(bundle_data, tf)
            tf.flush()
            os.fsync(tf.fileno())
        os.chmod(temp_name, 0o644)
        os.replace(temp_name, target_path)
        temp_name = None
    finally:
        if temp_name and os.path.exists(temp_name):
            try:
                os.remove(temp_name)
            except OSError:
                logger.warning("Gagal membersihkan file sementara: %s", temp_name)

def _model_needs_retrain(meta: Dict[str, Any]) -> bool:
    trained_at_str = meta.get("trained_at")
    if not trained_at_str:
        return True
    try:
        trained_at = datetime.fromisoformat(trained_at_str)
        if trained_at.tzinfo is None:
            trained_at = trained_at.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc) - trained_at > timedelta(days=CONFIG.MODEL_MAX_AGE_DAYS)
    except Exception:
        return True

def get_cached_bundle(bundle_path: Path) -> Tuple[Optional[Any], Optional[Dict[str, Any]], bool]:
    key = str(bundle_path)
    if not bundle_path.exists():
        _MODEL_CACHE.pop(key, None)
        return None, None, True

    current_mtime = bundle_path.stat().st_mtime_ns
    cached = _MODEL_CACHE.get(key)

    if cached and cached.get("mtime_ns") == current_mtime:
        bundle = cached["bundle"]
    else:
        try:
            bundle = joblib.load(bundle_path)
        except Exception as exc:
            logger.error("Gagal memuat bundle %s: %s", bundle_path, exc)
            return None, None, True
        meta_loaded = bundle.get("meta", {})
        if meta_loaded.get("sklearn_version") != sklearn.__version__:
            logger.warning(
                "Versi scikit-learn berbeda (model=%s, runtime=%s).",
                meta_loaded.get("sklearn_version"), sklearn.__version__
            )
        _MODEL_CACHE[key] = {"bundle": bundle, "mtime_ns": current_mtime}

    meta = bundle.get("meta", {})
    return bundle.get("pipeline"), meta, _model_needs_retrain(meta)


# ============================================================================
# 2. FEATURE ENGINEERING & VALIDASI DATA TOKO RINA
# ============================================================================

def compute_context_stats(df_train: pd.DataFrame) -> Dict[str, Any]:
    counts = df_train.groupby(COMBO_COLS).size()
    return {
        "n": int(len(df_train)),
        "combo_counts": {tuple(k): int(v) for k, v in counts.items()},
        "customer_median_total": {k: float(v) for k, v in df_train.groupby("customer_code")["total"].median().items()},
        "global_median_total": float(df_train["total"].median()),
    }

def add_context_features(df: pd.DataFrame, stats: Dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    n = max(stats["n"], 1)
    counts = stats["combo_counts"]
    keys = list(zip(out["customer_code"], out["payment_profile"], out["day_type"]))
    out["combo_logfreq"] = [np.log((counts.get(k, 0) + 0.5) / n) for k in keys]

    med = out["customer_code"].map(stats["customer_median_total"]).fillna(stats["global_median_total"])
    out["amount_ratio"] = np.log1p(out["total"]) - np.log1p(med)
    return out

def build_tx_pipeline(random_state: int = 42) -> Pipeline:
    total_pipeline = Pipeline([
        ("log1p", FunctionTransformer(np.log1p, validate=True)),
        ("scaler", StandardScaler()),
    ])
    cat_cols = ["day_type", "payment_profile"] + (["customer_code"] if CONFIG.USE_ONEHOT_CATEGORICAL else [])
    preprocessor = ColumnTransformer(
        transformers=[
            ("total_t", total_pipeline, ["total"]),
            ("credit_r_t", StandardScaler(), ["credit_ratio"]),
            ("ratio_t", StandardScaler(), ["amount_ratio"]),
            ("combo_t", StandardScaler(), ["combo_logfreq"]),
            ("cat_t", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        remainder="drop",
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("isoforest", IsolationForest(n_estimators=100, random_state=random_state)),
    ])

def validate_transaction_payload(transactions: List[Dict[str, Any]]) -> pd.DataFrame:
    if not transactions:
        raise ValueError("Payload transaksi kosong.")

    df = pd.DataFrame(transactions)
    missing = [c for c in REQUIRED_TX_FIELDS if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom wajib tidak lengkap: {missing}")

    if df[REQUIRED_TX_FIELDS].isnull().any().any():
        raise ValueError(f"Ditemukan null pada kolom wajib: {df[REQUIRED_TX_FIELDS].isnull().sum().to_dict()}")

    for col in ["cash", "transfer", "credit"]:
        if col not in df.columns:
            df[col] = 0.0

    df["total"] = pd.to_numeric(df["total"], errors="coerce")
    df["cash"] = pd.to_numeric(df["cash"], errors="coerce").fillna(0.0)
    df["transfer"] = pd.to_numeric(df["transfer"], errors="coerce").fillna(0.0)
    df["credit"] = pd.to_numeric(df["credit"], errors="coerce").fillna(0.0)

    if df["total"].isnull().any():
        raise ValueError("Kolom 'total' harus numerik valid.")
    if (df["total"] < 0.0).any():
        raise ValueError("Ada nominal total negatif. Transaksi retur harus diproses terpisah.")

    parsed_dates = pd.to_datetime(df["date"], errors="coerce")
    if parsed_dates.isnull().any():
        raise ValueError("Format kolom 'date' tidak dapat diparse.")

    df["parsed_date"] = parsed_dates
    df["day_type"] = np.where(parsed_dates.dt.dayofweek >= 5, "WEEKEND", "WEEKDAY")
    df["credit_ratio"] = np.where(df["total"] > 0, df["credit"] / df["total"], 0.0)

    def get_profile(row):
        c, t, cr = row["cash"] > 0, row["transfer"] > 0, row["credit"] > 0
        if c and not t and not cr:
            return "CASH_ONLY"
        if t and not c and not cr:
            return "TF_ONLY"
        if cr and not c and not t:
            return "CREDIT_ONLY"
        return "SPLIT_PAYMENT"

    df["payment_profile"] = df.apply(get_profile, axis=1)
    df["customer_code"] = df["customer_code"].astype(str).str.upper().str.strip()

    return df.sort_values("parsed_date", kind="stable").reset_index(drop=True)

def _df_to_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    out = df.copy()
    if "parsed_date" in out.columns:
        out["date"] = out["parsed_date"].dt.strftime("%Y-%m-%d")
        out = out.drop(columns=["parsed_date"])
    return out.to_dict("records")


# ============================================================================
# 3. BASELINES STATISTIK & ATURAN HEURISTIK
# ============================================================================

def calculate_amount_baselines(historical_amounts: np.ndarray) -> Dict[str, Any]:
    if len(historical_amounts) < 4:
        return {"ready": False}
    q25, q75 = np.percentile(historical_amounts, [25, 75])
    iqr = q75 - q25
    return {
        "ready": True,
        "mean": float(np.mean(historical_amounts)),
        "std": float(np.std(historical_amounts)),
        "iqr_lower": float(q25 - 1.5 * iqr),
        "iqr_upper": float(q75 + 1.5 * iqr),
    }

def evaluate_multivariate_heuristic_baseline(
    rec: Dict[str, Any],
    amount_baseline: Dict[str, Any],
    known_customers: Optional[set] = None,
    known_combos: Optional[set] = None,
) -> Dict[str, bool]:
    tot = float(rec["total"])
    cust = str(rec["customer_code"]).upper()
    credit = float(rec.get("credit", 0.0))

    iqr_flag = z_flag = False
    if amount_baseline.get("ready"):
        iqr_flag = bool(tot < amount_baseline["iqr_lower"] or tot > amount_baseline["iqr_upper"])
        std = amount_baseline["std"]
        z = (tot - amount_baseline["mean"]) / std if std > 0 else 0.0
        z_flag = bool(abs(z) > CONFIG.Z_SCORE_THRESHOLD)

    # Deteksi utang berisiko tinggi (High Risk Credit)
    if known_customers:
        unknown_cust_flag = cust not in known_customers
        high_risk_credit = unknown_cust_flag and (credit > 1_000_000.0)
    else:
        # Saat model belum terlatih (fallback), utang besar > Rp 1.000.000 langsung ditandai
        unknown_cust_flag = False
        high_risk_credit = credit > 1_000_000.0

    combo = (cust, str(rec.get("payment_profile", "")), str(rec.get("day_type", "")))
    unseen_combo_flag = bool(known_combos) and (combo not in known_combos)

    return {
        "iqr_flag": iqr_flag,
        "z_score_flag": z_flag,
        "unknown_cust_flag": unknown_cust_flag,
        "high_risk_credit": high_risk_credit,
        "unseen_combo_flag": unseen_combo_flag,
        "heuristic_multivariate_flag": bool(iqr_flag or high_risk_credit or unseen_combo_flag),
    }


# ============================================================================
# 4. AUDIT KAS FISIK
# ============================================================================

def train_cash_discrepancy_model(
    historical_discrepancies: List[float],
    min_required_days: int = CONFIG.MIN_CASH_DAYS,
) -> Dict[str, Any]:
    if len(historical_discrepancies) < min_required_days:
        return {
            "status": "INSUFFICIENT_DATA",
            "message": f"Data baru {len(historical_discrepancies)} hari (minimal {min_required_days}).",
        }

    arr = np.asarray(historical_discrepancies, dtype=float)
    zero_ratio = float(np.mean(arr == 0.0))
    is_degenerate = zero_ratio > CONFIG.DEGENERATE_ZERO_RATIO

    non_zero_abs = np.abs(arr[arr != 0.0])
    non_zero_tolerance = (
        float(np.percentile(non_zero_abs, 95.0)) if len(non_zero_abs) >= 10 else CONFIG.CASH_HARD_LIMIT
    )

    pipeline: Optional[Pipeline] = None
    if not is_degenerate:
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("ocsvm", OneClassSVM(kernel="rbf", gamma="scale", nu=0.03)),
        ])
        pipeline.fit(arr.reshape(-1, 1))

    meta = {
        "model_type": "EmpiricalRule" if is_degenerate else "OneClassSVM",
        "sklearn_version": sklearn.__version__,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_samples": int(len(arr)),
        "is_degenerate": is_degenerate,
        "zero_ratio": round(zero_ratio, 3),
        "score_threshold": 0.0,
        "non_zero_tolerance": non_zero_tolerance,
    }
    save_bundle_atomic({"pipeline": pipeline, "meta": meta}, CASH_BUNDLE_PATH)
    return {"status": "TRAINED", "metadata": meta}

def infer_cash_discrepancy_ml(current_discrepancy: float) -> Dict[str, Any]:
    pipeline, meta, needs_retrain = get_cached_bundle(CASH_BUNDLE_PATH)
    amount_txt = f"Rp {current_discrepancy:,.2f}"
    hard = CONFIG.CASH_HARD_LIMIT
    exceeds_hard = abs(current_discrepancy) > hard

    base = {
        "discrepancy": current_discrepancy,
        "exceeds_hard_limit": exceeds_hard,
        "hard_limit": hard,
    }

    if not meta:
        return {
            **base,
            "status": "FALLBACK_RULE",
            "is_anomaly": exceeds_hard,
            "needs_retrain": False,
            "message": (
                f"Selisih kas {amount_txt} melebihi batas keras Rp {hard:,.2f} (aturan fallback)."
                if exceeds_hard else
                f"Selisih kas {amount_txt} dalam batas aman Rp {hard:,.2f}."
            ),
        }

    if meta.get("is_degenerate") or pipeline is None:
        tol = meta.get("non_zero_tolerance", hard)
        is_anomaly = current_discrepancy != 0.0 and abs(current_discrepancy) > tol
        return {
            **base,
            "status": "EMPIRICAL_RULE",
            "is_anomaly": bool(is_anomaly),
            "threshold_used": tol,
            "needs_retrain": needs_retrain,
            "message": (
                f"Selisih kas {amount_txt} melebihi toleransi empiris Rp {tol:,.2f}."
                if is_anomaly else
                f"Selisih kas {amount_txt} dalam toleransi empiris riwayat."
            ),
        }

    score = float(pipeline.decision_function(np.array([[current_discrepancy]], dtype=float))[0])
    threshold = meta["score_threshold"]
    is_anomaly = bool(score < threshold)
    return {
        **base,
        "status": "ML_MODEL",
        "is_anomaly": is_anomaly,
        "ml_score": round(score, 4),
        "threshold": round(threshold, 4),
        "needs_retrain": needs_retrain,
        "message": (
            f"Selisih kas {amount_txt} anomali (skor OCSVM {score:.4f} < {threshold:.4f})."
            if is_anomaly else
            f"Selisih kas {amount_txt} wajar."
        ),
    }


# ============================================================================
# 5. AUDIT TRANSAKSI PENJUALAN TOKO RINA
# ============================================================================

def train_transaction_model(
    historical_transactions: List[Dict[str, Any]],
    min_required_samples: int = CONFIG.MIN_TX_SAMPLES,
) -> Dict[str, Any]:
    if len(historical_transactions) < min_required_samples:
        return {
            "status": "INSUFFICIENT_DATA",
            "message": f"Data baru {len(historical_transactions)} transaksi (minimal {min_required_samples}).",
        }

    df = validate_transaction_payload(historical_transactions)
    split_idx = int(len(df) * 0.8)
    df_train, df_val = df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()

    stats = compute_context_stats(df_train)
    pipeline = build_tx_pipeline(random_state=42)
    pipeline.fit(add_context_features(df_train, stats))

    val_scores = pipeline.decision_function(add_context_features(df_val, stats))
    threshold = float(np.percentile(val_scores, CONFIG.TX_FLAG_PERCENTILE))
    spread = float(np.std(val_scores))
    extreme_threshold = float(threshold - CONFIG.EXTREME_SPREAD_MULT * spread)

    meta = {
        "model_type": "IsolationForest",
        "sklearn_version": sklearn.__version__,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_samples": int(len(df_train)),
        "calibration_samples": int(len(df_val)),
        "score_threshold": threshold,
        "extreme_threshold": extreme_threshold,
        "use_onehot_categorical": CONFIG.USE_ONEHOT_CATEGORICAL,
        "context_stats": stats,
        "amount_baseline": calculate_amount_baselines(df_train["total"].values),
        "known_customers": sorted(df_train["customer_code"].unique().tolist()),
    }
    save_bundle_atomic({"pipeline": pipeline, "meta": meta}, TX_BUNDLE_PATH)
    return {"status": "TRAINED", "metadata": meta}

def _rule_reasons(base: Dict[str, bool]) -> List[str]:
    reasons = []
    if base.get("high_risk_credit"):
        reasons.append("utang besar oleh pelanggan baru")
    if base.get("unknown_cust_flag"):
        reasons.append("pelanggan baru")
    if base.get("unseen_combo_flag"):
        reasons.append("perubahan kebiasaan bayar/hari transaksi")
    if base.get("iqr_flag"):
        reasons.append("total transaksi di luar batas IQR")
    return reasons

def infer_transaction_anomalies_ml(
    daily_transactions: List[Dict[str, Any]],
    top_n: int = 5,
) -> Dict[str, Any]:
    pipeline, meta, needs_retrain = get_cached_bundle(TX_BUNDLE_PATH)
    df_daily = validate_transaction_payload(daily_transactions)
    records = _df_to_records(df_daily)

    model_ready = bool(pipeline is not None and meta and "context_stats" in meta)

    if model_ready:
        scores = pipeline.decision_function(add_context_features(df_daily, meta["context_stats"]))
        threshold = meta["score_threshold"]
        extreme_threshold = meta["extreme_threshold"]
        amount_base = meta["amount_baseline"]
        known_customers: Optional[set] = set(meta.get("known_customers", []))
        known_combos: Optional[set] = set(meta["context_stats"]["combo_counts"].keys())
    else:
        # Fallback Heuristik ketika model belum terlatih
        amount_base = calculate_amount_baselines(df_daily["total"].values)
        flagged_fb = []
        high_fb_count = 0
        breakdown_fb = {"high_risk_credit": 0, "unknown_customer": 0, "ml_extreme": 0}

        for rec in records:
            res = evaluate_multivariate_heuristic_baseline(rec, amount_base, None, None)
            if res["heuristic_multivariate_flag"]:
                is_high = bool(res.get("high_risk_credit", False))
                high_fb_count += int(is_high)
                breakdown_fb["high_risk_credit"] += int(is_high)
                breakdown_fb["unknown_customer"] += int(res.get("unknown_cust_flag", False))

                reasons = _rule_reasons(res)
                flagged_fb.append({
                    "transaction_data": rec,
                    "severity": "HIGH" if is_high else "REVIEW",
                    "flagged_by": ["aturan fallback: " + ", ".join(reasons)],
                    "ml_score": None,
                    "baseline_comparison": res,
                    "flagged_by_ml_only": False,
                    "is_extreme": False,
                })

        flagged_fb.sort(key=lambda x: x["severity"] != "HIGH")
        return {
            "status": "FALLBACK_HEURISTIC",
            "total_evaluated": len(records),
            "total_anomalies_found": len(flagged_fb),
            "high_severity_count": high_fb_count,
            "review_count": len(flagged_fb) - high_fb_count,
            "high_breakdown": breakdown_fb,
            "extreme_anomalies_count": 0,
            "needs_retrain": False,
            "anomalies": flagged_fb[:top_n],
        }

    # Model Ready: Evaluasi ML + Aturan
    flagged: List[Dict[str, Any]] = []
    breakdown = {"high_risk_credit": 0, "unknown_customer": 0, "ml_extreme": 0}
    high_count = review_count = 0

    for idx, rec in enumerate(records):
        base = evaluate_multivariate_heuristic_baseline(rec, amount_base, known_customers, known_combos)
        rule_flag = base["heuristic_multivariate_flag"]

        score = float(scores[idx])
        ml_flag = bool(score < threshold)
        ml_extreme = bool(score < extreme_threshold)

        if not (rule_flag or ml_flag):
            continue

        is_high = base["high_risk_credit"] or ml_extreme
        breakdown["high_risk_credit"] += int(base["high_risk_credit"])
        breakdown["unknown_customer"] += int(base["unknown_cust_flag"])
        breakdown["ml_extreme"] += int(ml_extreme)
        high_count += int(is_high)
        review_count += int(not is_high)

        flagged_by = []
        reasons = _rule_reasons(base)
        if reasons:
            flagged_by.append("aturan: " + ", ".join(reasons))
        if ml_flag:
            flagged_by.append("ML: " + ("skor ekstrem" if ml_extreme else "outlier"))

        flagged.append({
            "transaction_data": rec,
            "severity": "HIGH" if is_high else "REVIEW",
            "flagged_by": flagged_by,
            "ml_score": round(score, 4),
            "baseline_comparison": base,
            "flagged_by_ml_only": bool(ml_flag and not rule_flag),
            "is_extreme": ml_extreme,
        })

    flagged.sort(key=lambda x: (x["severity"] != "HIGH", x["ml_score"] if x["ml_score"] is not None else 0.0))

    return {
        "status": "SUCCESS",
        "total_evaluated": len(records),
        "total_anomalies_found": len(flagged),
        "high_severity_count": high_count,
        "review_count": review_count,
        "high_breakdown": breakdown,
        "extreme_anomalies_count": breakdown["ml_extreme"],
        "needs_retrain": bool(needs_retrain),
        "anomalies": flagged[:top_n],
    }


# ============================================================================
# 6. BENCHMARK MULTI-SEED TOKO RINA
# ============================================================================

def _mean_std(values: List[float]) -> str:
    return f"{float(np.mean(values)):.3f} ± {float(np.std(values)):.3f}"

def _combined_score(recall: float, fpr: float, prevalence: float) -> Tuple[float, float]:
    tp, fp = recall * prevalence, fpr * (1.0 - prevalence)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, f1

def benchmark_ml_vs_baselines(
    normal_transactions: List[Dict[str, Any]],
    seeds: Optional[List[int]] = None,
) -> Dict[str, Any]:
    seeds = seeds or [42, 123, 999]
    if len(normal_transactions) < CONFIG.MIN_BENCH_SAMPLES:
        return {"status": "INSUFFICIENT_DATA", "message": f"Dibutuhkan minimal {CONFIG.MIN_BENCH_SAMPLES} transaksi."}

    df_all = validate_transaction_payload(normal_transactions)
    n = len(df_all)
    train_end, val_end = int(n * 0.6), int(n * 0.8)
    df_train = df_all.iloc[:train_end].copy()
    df_val = df_all.iloc[train_end:val_end].copy()
    df_test_clean = df_all.iloc[val_end:].copy()

    customers = sorted(df_train["customer_code"].unique())
    known_customers = set(customers)
    existing_combos = set(zip(df_train["customer_code"], df_train["payment_profile"], df_train["day_type"]))

    profiles = ["CASH_ONLY", "TF_ONLY", "CREDIT_ONLY", "SPLIT_PAYMENT"]
    day_types = ["WEEKDAY", "WEEKEND"]
    all_combos = [(c, p, d) for c in customers for p in profiles for d in day_types]
    novel_combos = [c for c in all_combos if c not in existing_combos]

    median_tot = float(df_train["total"].median())
    q75_tot = float(np.percentile(df_train["total"], 75))

    stats = compute_context_stats(df_train)
    df_train_f = add_context_features(df_train, stats)
    df_val_f = add_context_features(df_val, stats)
    base_params = calculate_amount_baselines(df_train["total"].values)

    methods_names = ["ml", "heuristic", "iqr"]
    recall_rec = {m: {t: [] for t in ANOMALY_TYPES} for m in methods_names}
    fpr_rec = {m: [] for m in methods_names}
    auc_rec: List[float] = []
    n_by_type: Dict[str, int] = {}

    for seed in seeds:
        rng = np.random.default_rng(seed)
        pipeline = build_tx_pipeline(random_state=seed)
        pipeline.fit(df_train_f)
        threshold = float(np.percentile(pipeline.decision_function(df_val_f), CONFIG.TX_FLAG_PERCENTILE))

        def pick(seq):
            return seq[int(rng.integers(0, len(seq)))]

        rows: List[Dict[str, Any]] = []
        k = CONFIG.BENCH_SAMPLES_PER_TYPE
        base_date = "2026-09-01"

        # (a) Nominal Ekstrem
        for _ in range(k):
            tot = float(q75_tot * rng.uniform(4, 8))
            rows.append({"date": base_date, "total": tot, "cash": tot, "transfer": 0.0, "credit": 0.0,
                         "customer_code": pick(customers), "anomaly_type": "a_extreme_total"})

        # (b) Utang Penuh Tidak Wajar
        for _ in range(k):
            tot = float(median_tot * rng.uniform(1.5, 2.5))
            rows.append({"date": base_date, "total": tot, "cash": 0.0, "transfer": 0.0, "credit": tot,
                         "customer_code": pick(customers), "anomaly_type": "b_unusual_credit_ratio"})

        # (c) Kombinasi Cara Bayar Baru
        if novel_combos:
            take = min(k, len(novel_combos))
            for i in rng.permutation(len(novel_combos))[:take]:
                c, p, _ = novel_combos[int(i)]
                tot = float(median_tot * rng.uniform(0.9, 1.1))
                cr = tot if p == "CREDIT_ONLY" else 0.0
                rows.append({"date": base_date, "total": tot, "cash": tot - cr, "transfer": 0.0, "credit": cr,
                             "customer_code": c, "anomaly_type": "c_novel_combo"})

        # (d) Profil Langka
        for _ in range(k):
            tot = float(median_tot * 1.8)
            rows.append({"date": "2026-09-06", "total": tot, "cash": 0.0, "transfer": tot, "credit": 0.0,
                         "customer_code": pick(customers), "anomaly_type": "d_rare_profile"})

        df_synth = validate_transaction_payload(rows)
        df_test = pd.concat([df_test_clean.assign(anomaly_type="normal"), df_synth], ignore_index=True)
        atype = df_test["anomaly_type"].values
        y_true = (atype != "normal").astype(int)
        normal_mask = atype == "normal"

        scores = pipeline.decision_function(add_context_features(df_test.drop(columns=["anomaly_type"]), stats))
        preds = {"ml": (scores < threshold).astype(int)}

        heur, iqr = [], []
        for rec in df_test.to_dict("records"):
            r = evaluate_multivariate_heuristic_baseline(rec, base_params, known_customers, existing_combos)
            heur.append(int(r["heuristic_multivariate_flag"]))
            iqr.append(int(r["iqr_flag"]))
        preds["heuristic"], preds["iqr"] = np.array(heur), np.array(iqr)

        for name, p in preds.items():
            fpr_rec[name].append(float(np.mean(p[normal_mask])))
            for t in ANOMALY_TYPES:
                mask = atype == t
                n_by_type[t] = int(mask.sum())
                if mask.any():
                    recall_rec[name][t].append(float(np.mean(p[mask])))

        auc_rec.append(float(roc_auc_score(y_true, -scores)))

    active_types = [t for t in ANOMALY_TYPES if n_by_type.get(t, 0) > 0]
    mean_recall = {m: {t: float(np.mean(recall_rec[m][t])) for t in active_types} for m in methods_names}
    macro_recall = {m: float(np.mean(list(mean_recall[m].values()))) for m in methods_names}
    mean_fpr = {m: float(np.mean(fpr_rec[m])) for m in methods_names}
    realistic = {m: _combined_score(macro_recall[m], mean_fpr[m], CONFIG.BENCH_ASSUMED_PREVALENCE) for m in methods_names}

    delta = realistic["ml"][1] - realistic["heuristic"][1]
    margin = CONFIG.BENCH_MARGIN
    prev = f"{CONFIG.BENCH_ASSUMED_PREVALENCE:.0%}"

    if delta > margin:
        overall = f"ML unggul atas heuristik pada prevalensi {prev} (F1 {realistic['ml'][1]:.3f} vs {realistic['heuristic'][1]:.3f})."
    elif delta < -margin:
        overall = f"Heuristik unggul atas ML pada prevalensi {prev} (F1 {realistic['heuristic'][1]:.3f} vs {realistic['ml'][1]:.3f})."
    else:
        overall = f"ML dan heuristik setara pada prevalensi {prev} (F1 {realistic['ml'][1]:.3f} vs {realistic['heuristic'][1]:.3f})."

    return {
        "status": "SUCCESS",
        "eval_seeds": seeds,
        "recall_by_type": {m: {t: _mean_std(recall_rec[m][t]) if t in active_types else "N/A" for t in ANOMALY_TYPES} for m in methods_names},
        "false_positives_per_100_normal": {m: _mean_std([v * 100 for v in fpr_rec[m]]) for m in methods_names},
        "macro_recall": {m: round(macro_recall[m], 3) for m in methods_names},
        f"implied_f1_at_{prev}": {m: round(realistic[m][1], 3) for m in methods_names},
        "ml_roc_auc": _mean_std(auc_rec),
        "conclusion": overall,
    }


# ============================================================================
# 7. ORKESTRATOR HARIAN & RETRAIN
# ============================================================================

_FULL_COVERAGE = {"ML_MODEL", "EMPIRICAL_RULE"}

def evaluate_daily_audit_risks(
    current_discrepancy: float,
    vault_expense: float,
    unmatched_transfers: Optional[List[Dict[str, Any]]],
    daily_transactions: Optional[List[Dict[str, Any]]],
    cash_hard_limit: float = 10000.0,
    vault_emergency_limit: float = 500000.0,
    model_max_age_days: int = 7
) -> Dict[str, Any]:
    unmatched_transfers = unmatched_transfers or []
    risk_flags: List[str] = []
    review_flags: List[str] = []

    # 1. Aturan bisnis dinamis Toko Rina
    if vault_expense > vault_emergency_limit:
        risk_flags.append(f"Pengeluaran brankas pembelian barang besar: Rp {vault_expense:,.2f} (Batas: Rp {vault_emergency_limit:,.2f})")
    if len(unmatched_transfers) > 3:
        risk_flags.append(f"Ada {len(unmatched_transfers)} mutasi bank CR belum cocok")

    # 2. Kas Fisik (Ambang Batas Dinamis)
    exceeds_hard = abs(current_discrepancy) > cash_hard_limit
    if exceeds_hard:
        risk_flags.append(f"Selisih kas fisik Rp {current_discrepancy:,.2f} melebihi batas batas keras Rp {cash_hard_limit:,.2f}.")

    # 3. Transaksi Penjualan (Isolation Forest)
    tx_coverage = "NO_TRANSACTIONS"
    tx_eval: Dict[str, Any] = {"status": "SKIPPED", "anomalies": []}

    if daily_transactions:
        try:
            tx_eval = infer_transaction_anomalies_ml(daily_transactions, top_n=5)
            tx_coverage = "ML_MODEL" if tx_eval["status"] == "SUCCESS" else "FALLBACK_HEURISTIC"
            
            high = tx_eval.get("high_severity_count", 0)
            if high > 0:
                risk_flags.append(f"{high} transaksi penjualan ditandai berisiko tinggi oleh Model AI.")
            if tx_eval.get("review_count", 0) > 3:
                review_flags.append(f"{tx_eval['review_count']} transaksi outlier disarankan untuk ditinjau.")
        except Exception as exc:
            tx_coverage = "ERROR"
            tx_eval = {"status": "ERROR", "error": str(exc), "anomalies": []}

    if risk_flags:
        overall = "ATTENTION_REQUIRED"
    elif review_flags:
        overall = "REVIEW_SUGGESTED"
    else:
        overall = "HEALTHY"

    return {
        "status": overall,
        "component_coverage": {"cash_audit": "EVALUATED", "transaction_audit": tx_coverage},
        "risk_flags": risk_flags,
        "review_flags": review_flags,
        "discrepancy": current_discrepancy,
        "cash_hard_limit_applied": cash_hard_limit,
        "vault_expense": vault_expense,
        "transaction_audit_detail": tx_eval,
    }

def retrain_models_if_needed(
    historical_discrepancies: List[float],
    historical_transactions: List[Dict[str, Any]],
    force: bool = False,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    _, cash_meta, cash_stale = get_cached_bundle(CASH_BUNDLE_PATH)
    if force or cash_meta is None or cash_stale:
        result["cash"] = train_cash_discrepancy_model(historical_discrepancies)
    else:
        result["cash"] = {"status": "UP_TO_DATE"}

    _, tx_meta, tx_stale = get_cached_bundle(TX_BUNDLE_PATH)
    tx_legacy = tx_meta is not None and "context_stats" not in tx_meta
    if force or tx_meta is None or tx_stale or tx_legacy:
        result["transactions"] = train_transaction_model(historical_transactions)
    else:
        result["transactions"] = {"status": "UP_TO_DATE"}

    return result

def _generate_demo_transactions(n_days: int = 35, per_day: int = 30, seed: int = 0) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    customers = ["CUST_A", "CUST_B", "CUST_C", "CUST_D"]
    rows = []
    for d in range(n_days):
        date_str = f"2026-09-{(d % 28) + 1:02d}"
        for _ in range(per_day):
            cust = customers[int(rng.integers(0, 4))]
            tot = float(np.round(rng.lognormal(11.0, 0.7), -3))
            method = ["CASH", "TF", "CREDIT"][int(rng.choice(3, p=[.6, .25, .15]))]
            c = tot if method == "CASH" else 0.0
            t = tot if method == "TF" else 0.0
            cr = tot if method == "CREDIT" else 0.0
            rows.append({
                "date": date_str,
                "customer_code": cust,
                "total": tot,
                "cash": c,
                "transfer": t,
                "credit": cr
            })
    return rows