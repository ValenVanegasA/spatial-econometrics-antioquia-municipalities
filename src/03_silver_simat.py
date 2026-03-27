# =============================================================
# SILVER — SIMAT: Cobertura Neta Educación Secundaria
# Antioquia 2015-2023
# =============================================================
# INPUT : data/bronze/simat_bronze.parquet
# OUTPUT: data/silver/simat_silver.parquet
#         una fila = un municipio-año
#         columna clave: cobertura_secundaria (%)
# =============================================================

import pandas as pd
import os
import logging
from datetime import datetime

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename=f"logs/03_silver_simat_{datetime.today().strftime('%Y%m%d')}.log",
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)

BRONZE_PATH = os.path.join("data", "bronze", "simat_bronze.parquet")
SILVER_PATH = os.path.join("data", "silver")
os.makedirs(SILVER_PATH, exist_ok=True)

if not os.path.exists(BRONZE_PATH):
    raise FileNotFoundError("❌ Ejecuta primero 02_bronze_simat.py")

df = pd.read_parquet(BRONZE_PATH)
print(f"✅ Bronze cargado: {df.shape}")

# =============================================================
# PASO 1 — Renombrar columnas
# =============================================================

df = df.rename(columns={
    "Departamento"   : "departamento",
    "Código Entidad" : "cod_mpio",
    "Entidad"        : "municipio",
    "Indicador"      : "indicador",
    "Dato Numérico"  : "cobertura_secundaria",
    "Año"            : "year"
})

print(f"✅ Columnas renombradas: {df.columns.tolist()}")

# =============================================================
# PASO 2 — Filtrar Antioquia
# =============================================================

print("\n📌 Valores únicos departamento (muestra):")
print(df["departamento"].value_counts().head(10).to_string())

n_antes = len(df)
df = df[df["departamento"].str.upper().str.strip() == "ANTIOQUIA"]
print(f"\n✅ Filtrado Antioquia: {len(df):,} filas (de {n_antes:,})")

# =============================================================
# PASO 3 — Filtrar años 2015-2023
# =============================================================

df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
df = df[df["year"].between(2015, 2023)]
df = df[df["cod_mpio"] != "05000"]
print(f"✅ Filtrado 2015-2023: {len(df):,} filas")
print(f"✅ Años disponibles: {sorted(df['year'].dropna().unique().tolist())}")

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
# PASO 5 — Limpiar
# =============================================================

# Eliminar columna indicador (ya no se necesita)
if "indicador" in df.columns:
    df = df.drop(columns=["indicador"])

# Convertir cobertura a numérico
df["cobertura_secundaria"] = (
    df["cobertura_secundaria"]
    .astype(str)
    .str.replace(",", ".", regex=False)
)
df["cobertura_secundaria"] = pd.to_numeric(df["cobertura_secundaria"], errors="coerce")
# Eliminar nulos en variables clave
df = df.dropna(subset=["cod_mpio", "year", "cobertura_secundaria"])
df = df.sort_values(["cod_mpio", "year"]).reset_index(drop=True)

# =============================================================
# PASO 6 — Guardar Silver
# =============================================================

df.to_parquet(os.path.join(SILVER_PATH, "simat_silver.parquet"), index=False)
df.to_csv(os.path.join(SILVER_PATH, "simat_silver.csv"), index=False, encoding="utf-8-sig")

# =============================================================
# PASO 7 — Reporte de calidad
# =============================================================

print("\n" + "="*55)
print("  REPORTE DE CALIDAD — SILVER SIMAT")
print("="*55)
print(f"  Observaciones        : {len(df):,}")
print(f"  Municipios únicos    : {df['cod_mpio'].nunique():,}")
print(f"  Años                 : {sorted(df['year'].dropna().unique().tolist())}")
print(f"  Nulos cobertura      : {df['cobertura_secundaria'].isna().sum():,}")
print(f"  Cobertura mínima     : {df['cobertura_secundaria'].min():.1f}%")
print(f"  Cobertura máxima     : {df['cobertura_secundaria'].max():.1f}%")
print(f"  Cobertura promedio   : {df['cobertura_secundaria'].mean():.1f}%")
print("="*55)
print("\n📌 Primeras filas:")
print(df.head(6).to_string())

logging.info(f"Silver SIMAT guardado: {df.shape}")
print("\n✅ SILVER SIMAT COMPLETADO")