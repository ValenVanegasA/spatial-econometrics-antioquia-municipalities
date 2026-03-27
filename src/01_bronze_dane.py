# =============================================================
# BRONZE — Valor Agregado Municipal DANE 2015-2024
# Carga cruda del Excel original sin transformar
# =============================================================

import pandas as pd
import os
import logging
from datetime import datetime

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename=f"logs/01_bronze_{datetime.today().strftime('%Y%m%d')}.log",
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)

RAW_PATH    = os.path.join("data", "raw", "PIB-VA_Mpal_2015-2024pr_Publ_02-09-2025_v2.xlsm")
BRONZE_PATH = os.path.join("data", "bronze")
os.makedirs(BRONZE_PATH, exist_ok=True)

# --- verificar archivo ---
if not os.path.exists(RAW_PATH):
    raise FileNotFoundError(f"❌ Archivo no encontrado: {RAW_PATH}\n   Ponlo en data/raw/")

print(f"✅ Archivo encontrado: {RAW_PATH}")

# --- cargar hoja principal (fila 4 = encabezados, datos desde fila 5) ---
df = pd.read_excel(
    RAW_PATH,
    sheet_name="PIB Mpal 2015-2024 Cons",
    header=3,          # fila 4 en Excel = índice 3 en Python
    engine="openpyxl"
)

print(f"✅ Shape cargado: {df.shape}")
print("\n📌 Columnas detectadas:")
for i, col in enumerate(df.columns):
    print(f"   [{i}] {col}")
print("\n📌 Primeras 3 filas:")
print(df.head(3).to_string())

# --- guardar bronze ---
df.to_parquet(os.path.join(BRONZE_PATH, "dane_bronze.parquet"), index=False)
df.to_csv(os.path.join(BRONZE_PATH, "dane_bronze.csv"), index=False, encoding="utf-8-sig")

logging.info(f"Bronze guardado: {df.shape}")
print("\n✅ BRONZE COMPLETADO")