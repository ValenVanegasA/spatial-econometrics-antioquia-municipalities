# =============================================================
# BRONZE — SIMAT: Cobertura Neta Educación Secundaria
# Fuente: TerriData DNP — TerriData_Dim4.xlsx
# =============================================================

import pandas as pd
import os
import logging
from datetime import datetime
from pathlib import Path

os.chdir(Path(__file__).resolve().parents[2])
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename=f"logs/01_bronze_simat_{datetime.today().strftime('%Y%m%d')}.log",
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)

RAW_PATH    = os.path.join("data", "raw", "TerriData_Dim4.xlsx")
BRONZE_PATH = os.path.join("data", "bronze")
os.makedirs(BRONZE_PATH, exist_ok=True)

if not os.path.exists(RAW_PATH):
    raise FileNotFoundError(f"❌ Archivo no encontrado: {RAW_PATH}")

print(f"✅ Archivo encontrado: {RAW_PATH}")

# --- ver hojas disponibles ---
xl = pd.ExcelFile(RAW_PATH, engine="openpyxl")
print(f"\n📌 Hojas disponibles: {xl.sheet_names}")

# --- cargar primera hoja ---
df = pd.read_excel(RAW_PATH, sheet_name=0, engine="openpyxl")

print(f"\n✅ Shape: {df.shape}")
print("\n📌 Columnas:")
for i, col in enumerate(df.columns):
    print(f"   [{i}] {col}")
print("\n📌 Primeras 3 filas:")
print(df.head(3).to_string())
print("\n📌 Valores únicos Indicador:")
if "Indicador" in df.columns:
    print(df["Indicador"].value_counts().head(10).to_string())
print("\n📌 Años disponibles:")
if "Año" in df.columns:
    print(sorted(df["Año"].dropna().unique().tolist()))

df.to_parquet(os.path.join(BRONZE_PATH, "simat_bronze.parquet"), index=False)
df.to_csv(os.path.join(BRONZE_PATH, "simat_bronze.csv"), index=False, encoding="utf-8-sig")

logging.info(f"Bronze SIMAT guardado: {df.shape}")
print("\n✅ BRONZE SIMAT COMPLETADO")