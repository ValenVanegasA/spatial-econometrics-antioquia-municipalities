# =============================================================
# GOLD — Panel final para econometría espacial
# Antioquia 2015-2024
# =============================================================
# Variables incluidas hasta ahora:
#   - va_total, ln_va_total     (DANE)
#   - homicidios                (Seguridad / INMLCF)
#   - cobertura_secundaria      (SIMAT / TerriData)
# Pendientes (descomenta cuando tengas el silver):
#   - idf                       (DNP — desempeño fiscal)
#   - desplazados               (UARIV)
#   - mortalidad_infantil       (DANE — Estadísticas Vitales)
# =============================================================

import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename=f"logs/gold_panel_{datetime.today().strftime('%Y%m%d')}.log",
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)

SILVER_PATH = os.path.join("data", "silver")
GOLD_PATH   = os.path.join("data", "gold")
os.makedirs(GOLD_PATH, exist_ok=True)

# =============================================================
# PASO 1 — Cargar silvers
# =============================================================

dane = pd.read_parquet(os.path.join(SILVER_PATH, "dane_silver.parquet"))
seg  = pd.read_parquet(os.path.join(SILVER_PATH, "seguridad_silver.parquet"))
sim  = pd.read_parquet(os.path.join(SILVER_PATH, "simat_silver.parquet"))

print(f"✅ Silver DANE      : {dane.shape}")
print(f"✅ Silver Seguridad : {seg.shape}")
print(f"✅ Silver SIMAT     : {sim.shape}")

# =============================================================
# PASO 2 — Panel base: DANE es el ancla
# =============================================================

panel = dane[["cod_mpio", "municipio", "subregion", "year",
              "va_total", "ln_va_total"]].copy()

# =============================================================
# PASO 3 — Merge seguridad (homicidios)
# =============================================================

panel = panel.merge(seg[["cod_mpio", "year", "homicidios"]],
                    on=["cod_mpio", "year"], how="left")
panel["homicidios"] = panel["homicidios"].fillna(0).astype(int)
print(f"\n✅ Merge seguridad: {panel.shape}")

# =============================================================
# PASO 4 — Merge SIMAT (cobertura secundaria)
# SIMAT llega hasta 2023 — 2024 queda NaN (rezago estadístico)
# =============================================================

panel = panel.merge(sim[["cod_mpio", "year", "cobertura_secundaria"]],
                    on=["cod_mpio", "year"], how="left")
n_nulos_cob = panel["cobertura_secundaria"].isna().sum()
print(f"✅ Merge SIMAT: {panel.shape}")
print(f"   Nulos cobertura (esperado ~125 por año 2024): {n_nulos_cob}")

# =============================================================
# PASO 5 — Fuentes pendientes
# =============================================================

# --- DNP: Índice de Desempeño Fiscal ---
# dnp = pd.read_parquet(os.path.join(SILVER_PATH, "dnp_silver.parquet"))
# panel = panel.merge(dnp[["cod_mpio","year","idf"]], on=["cod_mpio","year"], how="left")

# --- UARIV: Desplazamiento forzado ---
# uariv = pd.read_parquet(os.path.join(SILVER_PATH, "uariv_silver.parquet"))
# panel = panel.merge(uariv[["cod_mpio","year","desplazados"]], on=["cod_mpio","year"], how="left")

# --- DANE Vitales: Mortalidad infantil ---
# vitales = pd.read_parquet(os.path.join(SILVER_PATH, "vitales_silver.parquet"))
# panel = panel.merge(vitales[["cod_mpio","year","mortalidad_infantil"]], on=["cod_mpio","year"], how="left")

# =============================================================
# PASO 6 — Ordenar y verificar balance
# =============================================================

panel = panel.sort_values(["cod_mpio", "year"]).reset_index(drop=True)

n_mpios    = panel["cod_mpio"].nunique()
n_años     = panel["year"].nunique()
n_obs      = len(panel)
n_esperado = n_mpios * n_años

print(f"\n📊 BALANCE DEL PANEL:")
print(panel.groupby("year")["cod_mpio"].count().to_string())
print(f"\n   Municipios         : {n_mpios:,}")
print(f"   Años               : {n_años}")
print(f"   Observaciones      : {n_obs:,}")
print(f"   Esperadas (balance): {n_esperado:,}")
print(f"   Estado             : {'✅ BALANCEADO' if n_obs == n_esperado else '⚠️  DESBALANCEADO'}")

# =============================================================
# PASO 7 — Estadísticas descriptivas
# =============================================================

cols_desc = ["va_total", "ln_va_total", "homicidios", "cobertura_secundaria"]
print("\n" + "="*55)
print("  ESTADÍSTICAS DESCRIPTIVAS — GOLD PANEL")
print("="*55)
print(panel[cols_desc].describe().round(2).to_string())
print("="*55)

# =============================================================
# PASO 8 — Guardar Gold (3 formatos)
# =============================================================

panel.to_parquet(os.path.join(GOLD_PATH, "panel_gold.parquet"), index=False)
panel.to_csv(os.path.join(GOLD_PATH, "panel_gold.csv"), index=False, encoding="utf-8-sig")
panel.to_excel(os.path.join(GOLD_PATH, "panel_gold.xlsx"), index=False)

print(f"\n✅ Gold parquet : data/gold/panel_gold.parquet")
print(f"✅ Gold CSV     : data/gold/panel_gold.csv")
print(f"✅ Gold Excel   : data/gold/panel_gold.xlsx")

logging.info(f"Gold panel guardado: {panel.shape} — cols: {panel.columns.tolist()}")
print("\n✅ GOLD COMPLETADO — PANEL LISTO PARA ECONOMETRÍA")