# =============================================================
# SILVER — Valor Agregado Municipal DANE 2015-2024
# Limpieza, estandarización DIVIPOLA, formato panel largo
# =============================================================
# ESTRUCTURA DEL EXCEL (confirmada):
#   Col A  → Año
#   Col E  → Código municipio (DIVIPOLA 5 dígitos)
#   Col F  → Subregión
#   Col G  → Municipio
#   Col H–V → Ramas económicas (VA por sector)
#   Col W  → VA Total
# =============================================================

import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename=f"logs/02_silver_{datetime.today().strftime('%Y%m%d')}.log",
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)

BRONZE_PATH = os.path.join("data", "bronze", "dane_bronze.parquet")
SILVER_PATH = os.path.join("data", "silver")
os.makedirs(SILVER_PATH, exist_ok=True)

if not os.path.exists(BRONZE_PATH):
    raise FileNotFoundError("❌ Ejecuta primero 01_bronze_dane.py")

df = pd.read_parquet(BRONZE_PATH)
print(f"✅ Bronze cargado: {df.shape}")
print("Columnas:", df.columns.tolist())

# =============================================================
# PASO 1 — Renombrar columnas clave
# Ajusta los nombres entre comillas si el Excel los tiene diferente
# (corre el bronze primero y mira exactamente cómo se llaman)
# =============================================================

# Detectar columna de año (columna A → índice 0)
col_year    = df.columns[0]
col_codmpio = df.columns[4]   # columna E
col_subr    = df.columns[5]   # columna F
col_mpio    = df.columns[6]   # columna G
col_va_total= df.columns[22]  # columna W (índice 22)

# Ramas económicas: columnas H a V = índices 7 a 21
cols_ramas  = df.columns[7:22].tolist()

print(f"\n📌 Columna año        : {col_year}")
print(f"📌 Columna cod_mpio   : {col_codmpio}")
print(f"📌 Columna subregion  : {col_subr}")
print(f"📌 Columna municipio  : {col_mpio}")
print(f"📌 Columna VA total   : {col_va_total}")
print(f"📌 Ramas económicas   : {cols_ramas}")

# =============================================================
# PASO 2 — Seleccionar y renombrar
# =============================================================

cols_usar = [col_year, col_codmpio, col_subr, col_mpio, col_va_total] + cols_ramas

df = df[cols_usar].copy()

rename = {
    col_year    : "year",
    col_codmpio : "cod_mpio",
    col_subr    : "subregion",
    col_mpio    : "municipio",
    col_va_total: "va_total"
}
df = df.rename(columns=rename)

# Renombrar ramas con nombres limpios (rama_01, rama_02, ...)
ramas_rename = {col: f"rama_{str(i+1).zfill(2)}" for i, col in enumerate(cols_ramas)}
df = df.rename(columns=ramas_rename)
cols_ramas_limpias = list(ramas_rename.values())

print(f"\n✅ Columnas finales: {df.columns.tolist()}")

# =============================================================
# PASO 3 — Limpiar filas
# Eliminar filas sin código municipio (totales, subtotales, vacías)
# =============================================================

n_antes = len(df)
df = df[df["cod_mpio"].notna()]
df = df[~df["cod_mpio"].astype(str).str.lower().str.contains(
    "total|nacional|region|subtotal|código|nan", na=False)]
print(f"\n✅ Filas eliminadas (totales/nulos): {n_antes - len(df)}")

# =============================================================
# PASO 4 — Estandarizar DIVIPOLA a 5 dígitos
# =============================================================

df["cod_mpio"] = (
    df["cod_mpio"]
    .astype(str)
    .str.strip()
    .str.replace(r"\.0$", "", regex=True)
    .str.zfill(5)
)

print(f"✅ Ejemplo DIVIPOLA: {df['cod_mpio'].head(3).tolist()}")

# =============================================================
# PASO 5 — Tipos de datos
# =============================================================

df["year"]     = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
df["va_total"] = pd.to_numeric(df["va_total"], errors="coerce")

for col in cols_ramas_limpias:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["cod_mpio", "year", "va_total"])
df = df[df["year"].between(2015, 2024)]

# =============================================================
# PASO 6 — Variable dependiente: logaritmo del VA total
# =============================================================

df["ln_va_total"] = np.log(df["va_total"].clip(lower=0.001))

# =============================================================
# PASO 7 — Guardar Silver (panel: una fila = un municipio-año)
# =============================================================

df = df.sort_values(["cod_mpio", "year"]).reset_index(drop=True)

df.to_parquet(os.path.join(SILVER_PATH, "dane_silver.parquet"), index=False)
df.to_csv(os.path.join(SILVER_PATH, "dane_silver.csv"), index=False, encoding="utf-8-sig")

# =============================================================
# PASO 8 — Reporte de calidad
# =============================================================

print("\n" + "="*55)
print("  REPORTE DE CALIDAD — SILVER")
print("="*55)
print(f"  Observaciones     : {len(df):,}")
print(f"  Municipios únicos : {df['cod_mpio'].nunique():,}")
print(f"  Años              : {sorted(df['year'].dropna().unique().tolist())}")
print(f"  Nulos VA total    : {df['va_total'].isna().sum():,}")
print(f"  VA mínimo         : {df['va_total'].min():,.1f} millones COP")
print(f"  VA máximo         : {df['va_total'].max():,.1f} millones COP")
print("="*55)

logging.info(f"Silver guardado: {df.shape}")
print("\n✅ SILVER COMPLETADO")
