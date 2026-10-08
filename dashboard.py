"""
dashboard.py
Antarmuka Web Enterprise Toko Rina (Streamlit).
Layout Modular Tab Horizontal, Tabel Ber-scroll Internal, dan Filter Pencarian Cepat.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

# 1. Inisialisasi Database
from src.database import Base, SessionLocal, engine
Base.metadata.create_all(bind=engine)
db = SessionLocal()

from src.services.anomaly_ml_service import evaluate_daily_audit_risks
from src.services.ingestion_service import parse_bank_mutation_csv, parse_sales_report_pdf
from src.services.receivable_service import (
    get_all_debts_for_table,
    get_total_receivable_balance,
    record_daily_credit_sales,
)
from src.services.transfer_service import reconcile_daily_bank_mutations

st.set_page_config(
    page_title="Toko Rina | Enterprise Audit System",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling agar tabel compact dan scrollable
st.markdown("""
<style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    div[data-testid="stMetricValue"] { font-size: 1.8rem; }
</style>
""", unsafe_allow_html=True)

# Helper Riwayat Audit
HISTORY_FILE = Path("data/processed/upload_history.json")
def get_history() -> List[Dict[str, Any]]:
    if not HISTORY_FILE.exists():
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

# ============================================================================
# SIDEBAR: KONTROL INPUT
# ============================================================================
st.sidebar.title("📥 Panel Audit Harian")
uploaded_pdf = st.sidebar.file_uploader("1. Laporan Penjualan (PDF Toko Rina)", type=["pdf"])
uploaded_csv = st.sidebar.file_uploader("2. Mutasi Rekening Bank (CSV)", type=["csv"])

st.sidebar.markdown("---")
st.sidebar.subheader("💵 Fisik Kasir")
laci_input = st.sidebar.number_input("Uang Keluar Laci (Rp)", min_value=0.0, step=10000.0, value=0.0)
brankas_input = st.sidebar.number_input("Uang Masuk Brankas (Rp)", min_value=0.0, step=50000.0, value=0.0)
vault_expense = st.sidebar.number_input("Pengeluaran Brankas (Rp)", min_value=0.0, step=10000.0, value=0.0)

btn_audit = st.sidebar.button("🚀 Jalankan Audit & Rekonsiliasi", type="primary", use_container_width=True)

# ============================================================================
# LOGIKA EKSEKUSI AUDIT
# ============================================================================
if btn_audit:
    if not uploaded_pdf or not uploaded_csv:
        st.sidebar.error("⚠️ File PDF dan CSV wajib diunggah bersamaan!")
    else:
        with st.spinner("Memproses pencocokan data dan menyusun berkas arsip..."):
            pdf_data = parse_sales_report_pdf(uploaded_pdf)
            audit_date = pdf_data["audit_date"]
            today_sales = pdf_data["records"]
            bank_mutations = parse_bank_mutation_csv(uploaded_csv)

            folder_tgl = Path("data/raw") / audit_date
            folder_tgl.mkdir(parents=True, exist_ok=True)

            # 1 & 2. Simpan Kas Fisik
            f_laci = {"date": audit_date, "nominal": laci_input}
            (folder_tgl / "uang_keluar_laci.json").write_text(json.dumps(f_laci, indent=2), encoding="utf-8")

            f_brankas = {"date": audit_date, "nominal": brankas_input}
            (folder_tgl / "uang_masuk_brankas.json").write_text(json.dumps(f_brankas, indent=2), encoding="utf-8")

            # 3. Simpan Catatan Utang
            record_daily_credit_sales(db, audit_date, today_sales)

            # 4 & 5. Rekonsiliasi Bank CR/DB
            reconcile_res = reconcile_daily_bank_mutations(
                db=db,
                audit_date=audit_date,
                bank_mutations=bank_mutations,
                today_sales_records=today_sales
            )

            # Evaluasi Kas & ML
            total_cash_fisik = laci_input + brankas_input
            total_cash_pdf = sum(row.get("cash", 0.0) for row in today_sales)
            discrepancy = total_cash_fisik - total_cash_pdf

            unmatched_list = reconcile_res.get("unmatched_transfers_list", [])
            audit_eval = evaluate_daily_audit_risks(
                current_discrepancy=discrepancy,
                vault_expense=vault_expense,
                unmatched_transfers=unmatched_list,
                daily_transactions=today_sales
            )

            # Simpan Riwayat Sesi
            HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
            hist = [h for h in get_history() if h.get("audit_date") != audit_date]
            hist.insert(0, {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "audit_date": audit_date,
                "pdf_file": uploaded_pdf.name,
                "csv_file": uploaded_csv.name,
                "cash_collected": total_cash_fisik,
                "cash_discrepancy": discrepancy,
                "debt_settled_count": reconcile_res.get("total_debt_settled", 0),
                "audit_status": audit_eval.get("status", "UNKNOWN"),
            })
            HISTORY_FILE.write_text(json.dumps(hist, indent=2), encoding="utf-8")

            st.session_state["executed"] = True
            st.session_state["audit_date"] = audit_date
            st.session_state["folder_tgl"] = str(folder_tgl)
            st.session_state["eval"] = audit_eval
            st.session_state["discrepancy"] = discrepancy
            st.session_state["cash_fisik"] = total_cash_fisik
            st.session_state["bank_mutations"] = bank_mutations
            st.toast(f"Audit {audit_date} selesai!", icon="✅")

# ============================================================================
# STRUKTUR UTAMA DENGAN TAB HORIZONTAL
# ============================================================================
st.title("⚖️ Toko Rina - Automated Audit & Reconciliation")

tab_overview, tab_piutang, tab_arsip, tab_history = st.tabs([
    "📊 Ringkasan & Status Audit",
    "📋 Buku Piutang Pelanggan",
    "📂 Penjelajah 5 Berkas Arsip",
    "📑 Riwayat Pengunggahan"
])

# ----------------------------------------------------------------------------
# TAB 1: RINGKASAN & STATUS AUDIT
# ----------------------------------------------------------------------------
with tab_overview:
    # 3 Metrik Cards
    c1, c2, c3 = st.columns(3)
    cash_val = st.session_state.get("cash_fisik", laci_input + brankas_input)
    disc_val = st.session_state.get("discrepancy", 0.0)

    b_mutations = st.session_state.get("bank_mutations", [])
    cr_sum = sum(m["amount"] for m in b_mutations if m.get("type") == "CR")
    db_sum = sum(m["amount"] for m in b_mutations if m.get("type") == "DB")
    net_tf = cr_sum - db_sum

    total_utang_aktif = get_total_receivable_balance(db)

    with c1:
        st.metric(
            label="💵 Card Cash (Brankas + Laci)",
            value=f"Rp {cash_val:,.0f}",
            delta=f"Selisih Kas: Rp {disc_val:,.0f}" if disc_val != 0.0 else "Kas Seimbang",
            delta_color="normal" if abs(disc_val) <= 10000.0 else "inverse"
        )
    with c2:
        st.metric(
            label="💳 Card Transfer (Net Mutasi Masuk)",
            value=f"Rp {net_tf:,.0f}",
            delta=f"-Rp {db_sum:,.0f} (DB Keluar)" if db_sum > 0 else "Nihil Pengeluaran",
            delta_color="off"
        )
    with c3:
        st.metric(
            label="📝 Card Credit (Total Utang Konsumen)",
            value=f"Rp {total_utang_aktif:,.0f}",
            delta="Tagihan Aktif",
            delta_color="off"
        )

    st.markdown("---")

    # Banner Evaluasi
    if st.session_state.get("executed"):
        eval_res = st.session_state["eval"]
        status = eval_res.get("status", "UNKNOWN")
        if "HEALTHY" in status:
            st.success(f"### Status Audit: {status}\nTidak ditemukan anomali. Kas seimbang dan mutasi sesuai.")
        elif status == "REVIEW_SUGGESTED":
            st.warning(f"### Status Audit: {status}\nPerhatian: Model AI menemukan kejanggalan pada pola transaksi.")
        else:
            st.error(f"### Status Audit: {status}\nPERINGATAN: Selisih kas fisik di luar batas toleransi atau mutasi bank tak cocok.")

        with st.expander("🔍 Lihat Detail Evaluasi Teknis AI Engine"):
            st.json(eval_res)
    else:
        st.info("💡 Unggah berkas di sidebar dan klik **Jalankan Audit** untuk memulai audit harian.")

    # Mutasi Pengeluaran DB Toko jika ada
    if db_sum > 0:
        st.subheader("📉 Riwayat Pengeluaran Bank (DB)")
        df_db = pd.DataFrame([m for m in b_mutations if m.get("type") == "DB"])
        df_db["amount"] = df_db["amount"].apply(lambda x: f"Rp {x:,.0f}")
        st.dataframe(df_db, use_container_width=True, height=200, hide_index=True)

# ----------------------------------------------------------------------------
# TAB 2: BUKU PIUTANG (DILENGKAPI PENCARIAN & KETINGGIAN TERKUNCI)
# ----------------------------------------------------------------------------
with tab_piutang:
    st.subheader("📋 Daftar Tagihan Pelanggan (Buku Piutang)")
    
    col_f1, col_f2 = st.columns([3, 1])
    with col_f1:
        search_cust = st.text_input("🔍 Cari Kode / Nama Pelanggan:", placeholder="Contoh: MALIBU, ATLANTIK, B5...")
    with col_f2:
        show_all = st.checkbox("Sertakan yang LUNAS", value=False)

    debts = get_all_debts_for_table(db, include_settled=show_all)
    if debts:
        df_debts = pd.DataFrame(debts)
        
        # Filter pencarian
        if search_cust:
            df_debts = df_debts[df_debts["Kode / Nama Pelanggan"].str.contains(search_cust.upper(), na=False)]

        df_debts["Total Bon Asli"] = df_debts["Total Bon Asli"].apply(lambda x: f"Rp {x:,.0f}")
        df_debts["Sisa Utang"] = df_debts["Sisa Utang"].apply(lambda x: f"Rp {x:,.0f}")
        
        # height=400 mengunci tabel agar scroll di dalam kontainer, tidak merusak layout halaman
        st.dataframe(
            df_debts,
            use_container_width=True,
            height=400,
            hide_index=True
        )
        st.caption(f"Menampilkan {len(df_debts)} entri tagihan.")
    else:
        st.success("🎉 Tidak ada utang aktif! Semua tagihan berstatus lunas.")

# ----------------------------------------------------------------------------
# TAB 3: PENJELAJAH 5 BERKAS ARSIP (BERDAMPINGAN DENGAN SELECTOR)
# ----------------------------------------------------------------------------
with tab_arsip:
    st.subheader("📂 Penjelajah Berkas JSON Hasil Audit")
    
    active_date = st.session_state.get("audit_date")
    if active_date:
        dir_path = Path("data/raw") / active_date
        st.caption(f"Folder Sumber: `{dir_path}`")

        c_select, c_view = st.columns([1, 2])
        
        file_map = {
            "1. Uang Keluar Laci": "uang_keluar_laci.json",
            "2. Uang Masuk Brankas": "uang_masuk_brankas.json",
            "3. Catatan Utang": "catatan_utang.json",
            "4. Hasil Matching Bank": "hasil_matching.json",
            "5. Mutasi DB Pengeluaran": "mutasi_db_bank.json"
        }

        with c_select:
            selected_label = st.radio("Pilih Berkas untuk Diperiksa:", list(file_map.keys()))
            selected_file = file_map[selected_label]
            target_file_path = dir_path / selected_file

        with c_view:
            if target_file_path.exists():
                file_content = json.loads(target_file_path.read_text(encoding="utf-8"))
                st.markdown(f"**Isi Berkas:** `{selected_file}`")
                
                # Tampilkan tabel jika datanya list, atau JSON viewer jika dict sederhana
                if isinstance(file_content, list) and len(file_content) > 0 and isinstance(file_content[0], dict):
                    st.dataframe(pd.DataFrame(file_content), use_container_width=True, height=350, hide_index=True)
                else:
                    st.json(file_content)

                st.download_button(
                    label=f"⬇️ Unduh {selected_file}",
                    data=json.dumps(file_content, indent=2),
                    file_name=selected_file,
                    mime="application/json"
                )
            else:
                st.warning(f"Berkas {selected_file} belum ditemukan di direktori sesi ini.")
    else:
        st.info("💡 Berkas JSON harian akan otomatis dibuat dan siap dijelajahi setelah Anda menjalankan audit di sidebar.")

# ----------------------------------------------------------------------------
# TAB 4: RIWAYAT SESI PENGUNGGAHAN
# ----------------------------------------------------------------------------
with tab_history:
    st.subheader("📑 Riwayat Pengunggahan & Sesi Audit Sebelumnya")
    history_data = get_history()
    if history_data:
        df_h = pd.DataFrame(history_data)
        df_h_disp = pd.DataFrame({
            "Waktu": df_h["timestamp"],
            "Tanggal Audit": df_h["audit_date"],
            "PDF Penjualan": df_h["pdf_file"],
            "CSV Mutasi": df_h["csv_file"],
            "Kas Fisik": df_h["cash_collected"].apply(lambda x: f"Rp {x:,.0f}"),
            "Selisih Kas": df_h["cash_discrepancy"].apply(lambda x: f"Rp {x:,.0f}"),
            "Utang Lunas": df_h["debt_settled_count"].apply(lambda x: f"{x} Pelanggan"),
            "Status AI": df_h["audit_status"]
        })
        st.dataframe(df_h_disp, use_container_width=True, height=350, hide_index=True)
    else:
        st.info("Belum ada riwayat audit yang tersimpan.")

db.close()