# =============================================================
# SILVER — Población Municipal DANE 2015-2035
# Estandarización DIVIPOLA, formato panel largo
# =============================================================

import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime
from pathlib import Path

os.chdir(Path(__file__).resolve().parents[2])
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename=f"logs/02_silver_poblacion_{datetime.today().strftime('%Y%m%d')}.log",
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)

BRONZE_PATH = os.path.join("data", "bronze", "poblacion_bronze.parquet")
SILVER_PATH = os.path.join("data", "silver")
os.makedirs(SILVER_PATH, exist_ok=True)

if not os.path.exists(BRONZE_PATH):
    raise FileNotFoundError(
        "❌ Ejecuta primero src/pipelines/01_bronze_dane.py"
    )

df = pd.read_parquet(BRONZE_PATH)
print(f"✅ Bronze cargado: {df.shape}")

# =============================================================
# PASO 1 — Transformar a formato panel (long format)
# =============================================================
# Seleccionar columnas de interes
years = [str(y) for y in range(2015, 2036)]
# Solo quedarse con las columnas que realmente existen
years_present = [y for y in years if y in df.columns]

cols_to_keep = ["DPMP", "MPIO"] + years_present
df = df[cols_to_keep].copy()

# Melt 
df_long = pd.melt(
    df, 
    id_vars=["DPMP", "MPIO"],
    value_vars=years_present,
    var_name="year",
    value_name="poblacion"
)

# =============================================================
# PASO 2 — Limpiar y formatear DIVIPOLA
# =============================================================
df_long.rename(columns={"DPMP": "cod_mpio", "MPIO": "municipio_poblacion"}, inplace=True)

# Limpiar cod_mpio
df_long = df_long[df_long["cod_mpio"].notna()]
df_long = df_long[~df_long["cod_mpio"].astype(str).str.lower().str.contains("total|dane", na=False)]

df_long["cod_mpio"] = (
    df_long["cod_mpio"]
    .astype(str)
    .str.strip()
    .str.replace(r"\.0$", "", regex=True)
    .str.zfill(5)
)

# =============================================================
# PASO 3 — Tipos de datos y filtro
# =============================================================
df_long["year"] = pd.to_numeric(df_long["year"], errors="coerce").astype("Int64")
df_long["poblacion"] = pd.to_numeric(df_long["poblacion"], errors="coerce")

df_long = df_long.dropna(subset=["cod_mpio", "year", "poblacion"])
# Opcional: restringimos el panel de poblacion al rango del proyeccto 2015-2024
df_long = df_long[df_long["year"].between(2015, 2024)]

df_long = df_long.sort_values(["cod_mpio", "year"]).reset_index(drop=True)

# =============================================================
# PASO 4 — Guardar Silver
# =============================================================
df_long.to_parquet(os.path.join(SILVER_PATH, "poblacion_silver.parquet"), index=False)
df_long.to_csv(os.path.join(SILVER_PATH, "poblacion_silver.csv"), index=False, encoding="utf-8-sig")

print("\n" + "="*55)
print("  REPORTE DE CALIDAD — SILVER POBLACION")
print("="*55)
print(f"  Observaciones     : {len(df_long):,}")
print(f"  Municipios únicos : {df_long['cod_mpio'].nunique():,}")
print(f"  Años              : {sorted(df_long['year'].dropna().unique().tolist())}")
print(f"  Nulos población   : {df_long['poblacion'].isna().sum():,}")
print(f"  Población mínima  : {df_long['poblacion'].min():,.0f}")
print(f"  Población máxima  : {df_long['poblacion'].max():,.0f}")
print("="*55)

logging.info(f"Silver Población guardado: {df_long.shape}")
print("\n✅ SILVER POBLACION COMPLETADO")
