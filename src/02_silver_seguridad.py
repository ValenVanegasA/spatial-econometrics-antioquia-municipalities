# =============================================================
# SILVER — Seguridad: Homicidios por municipio-año
# Antioquia 2015-2024
# =============================================================
# INPUT : data/bronze/seguridad_bronze.parquet
# OUTPUT: data/silver/seguridad_silver.parquet
#         una fila = un municipio-año
#         columna clave: homicidios (conteo total)
# =============================================================

import pandas as pd
import os
import logging
from datetime import datetime

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename=f"logs/02_silver_seguridad_{datetime.today().strftime('%Y%m%d')}.log",
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)

BRONZE_PATH = os.path.join("data", "bronze", "seguridad_bronze.parquet")
SILVER_PATH = os.path.join("data", "silver")
os.makedirs(SILVER_PATH, exist_ok=True)

if not os.path.exists(BRONZE_PATH):
    raise FileNotFoundError("❌ Ejecuta primero 01_bronze_seguridad.py")

df = pd.read_parquet(BRONZE_PATH)
print(f"✅ Bronze cargado: {df.shape}")

# =============================================================
# PASO 1 — Filtrar Antioquia
# =============================================================

n_antes = len(df)
df = df[df["DEPARTAMENTO"].str.upper().str.strip() == "ANTIOQUIA"]
print(f"✅ Filtrado Antioquia: {len(df):,} filas (de {n_antes:,})")

# =============================================================
# PASO 2 — Filtrar homicidios
# Revisa el bronze para confirmar el valor exacto en Tipo_Delito
# =============================================================

print("\n📌 Tipos de delito en Antioquia:")
print(df["Tipo_Delito"].value_counts().to_string())

df = df[df["Tipo_Delito"].str.strip() == "Homicidio-intencional"]
print(f"\n✅ Filtrado homicidios: {len(df):,} filas")

if len(df) == 0:
    print("\n⚠️  No se encontraron registros con Tipo_Delito == 'HOMICIDIOS'")
    print("   Revisa los valores únicos impresos arriba y ajusta el filtro")
    raise ValueError("Filtro homicidios vacío — ajusta el valor en Tipo_Delito")

# =============================================================
# PASO 3 — Extraer año de FECHA HECHO
# =============================================================

df["FECHA HECHO"] = pd.to_datetime(df["FECHA HECHO"], errors="coerce")
df["year"] = df["FECHA HECHO"].dt.year
df = df.dropna(subset=["year"])
df["year"] = df["year"].astype(int)

print(f"\n✅ Años disponibles: {sorted(df['year'].unique().tolist())}")

# =============================================================
# PASO 4 — Estandarizar DIVIPOLA a 5 dígitos
# =============================================================

df["cod_mpio"] = (
    df["CODIGO DANE"]
    .astype(str)
    .str.strip()
    .str.replace(r"\.0$", "", regex=True)
    .str.zfill(5)
)

print(f"✅ Ejemplo DIVIPOLA: {df['cod_mpio'].head(3).tolist()}")

# =============================================================
# PASO 5 — Filtrar 2015-2024
# =============================================================

df = df[df["year"].between(2015, 2024)]
print(f"✅ Filtrado 2015-2024: {len(df):,} filas")

# =============================================================
# PASO 6 — Agregar: homicidios por municipio-año
# =============================================================

df_agg = (
    df.groupby(["cod_mpio", "year"])["CANTIDAD"]
    .sum()
    .reset_index()
    .rename(columns={"CANTIDAD": "homicidios"})
)

df_agg["homicidios"] = df_agg["homicidios"].astype(int)
df_agg = df_agg.sort_values(["cod_mpio", "year"]).reset_index(drop=True)

print(f"\n✅ Panel homicidios: {df_agg.shape}")
print(f"   Municipios únicos: {df_agg['cod_mpio'].nunique()}")
print(f"   Años únicos      : {sorted(df_agg['year'].unique().tolist())}")
print("\n📌 Muestra:")
print(df_agg.head(10).to_string())

# =============================================================
# PASO 7 — Guardar Silver
# =============================================================

df_agg.to_parquet(os.path.join(SILVER_PATH, "seguridad_silver.parquet"), index=False)
df_agg.to_csv(os.path.join(SILVER_PATH, "seguridad_silver.csv"), index=False, encoding="utf-8-sig")

# =============================================================
# PASO 8 — Reporte de calidad
# =============================================================

print("\n" + "="*55)
print("  REPORTE DE CALIDAD — SILVER SEGURIDAD")
print("="*55)
print(f"  Observaciones     : {len(df_agg):,}")
print(f"  Municipios únicos : {df_agg['cod_mpio'].nunique():,}")
print(f"  Años              : {sorted(df_agg['year'].unique().tolist())}")
print(f"  Homicidios mínimo : {df_agg['homicidios'].min()}")
print(f"  Homicidios máximo : {df_agg['homicidios'].max()}")
print(f"  Homicidios total  : {df_agg['homicidios'].sum():,}")
print("="*55)

logging.info(f"Silver seguridad guardado: {df_agg.shape}")
print("\n✅ SILVER SEGURIDAD COMPLETADO")