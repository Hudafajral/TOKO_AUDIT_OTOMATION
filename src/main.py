"""
src/main.py
Pintu masuk FastAPI server audit Toko Rina.
"""
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from src.routers.web_router import router as web_router
from src.routers.audit_router import router as audit_api_router

app = FastAPI(title="Toko Rina Enterprise Audit WebApp")

# Pastikan direktori static dibuat otomatis jika belum ada
static_dir = Path("static")
static_dir.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")

# Daftarkan Router
app.include_router(web_router)
app.include_router(audit_api_router)