# =============================================================
# BRONZE — Seguridad (Delitos Colombia)
# Carga cruda del parquet original sin transformar
# =============================================================

import pandas as pd
import os
import logging
from datetime import datetime
from pathlib import Path

os.chdir(Path(__file__).resolve().parents[2])
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename=f"logs/01_bronze_seguridad_{datetime.today().strftime('%Y%m%d')}.log",
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)

RAW_PATH    = os.path.join("data", "raw", "delitos_colombia.parquet")
BRONZE_PATH = os.path.join("data", "bronze")
os.makedirs(BRONZE_PATH, exist_ok=True)

if not os.path.exists(RAW_PATH):
    raise FileNotFoundError(f"❌ Archivo no encontrado: {RAW_PATH}\n   Ponlo en data/raw/ — puede ser .parquet o .csv")

print(f"✅ Archivo encontrado: {RAW_PATH}")

df = pd.read_parquet(RAW_PATH)

print(f"✅ Shape: {df.shape}")
print("\n📌 Columnas:")
for i, col in enumerate(df.columns):
    print(f"   [{i}] {col}")
print("\n📌 Primeras 3 filas:")
print(df.head(3).to_string())
print("\n📌 Valores únicos DEPARTAMENTO (muestra):")
print(df["DEPARTAMENTO"].value_counts().head(10).to_string())
print("\n📌 Valores únicos Tipo_Delito:")
print(df["Tipo_Delito"].value_counts().to_string())

df.to_parquet(os.path.join(BRONZE_PATH, "seguridad_bronze.parquet"), index=False)
df.to_csv(os.path.join(BRONZE_PATH, "seguridad_bronze.csv"), index=False, encoding="utf-8-sig")

logging.info(f"Bronze seguridad guardado: {df.shape}")
print("\n✅ BRONZE SEGURIDAD COMPLETADO")