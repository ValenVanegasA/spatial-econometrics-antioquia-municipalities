# =============================================================
# GOLD — Panel final para econometría espacial
# =============================================================
# Por ahora solo tiene DANE (variable dependiente).
# Cuando descargues SIMAT, DNP, UARIV, etc., se agregan
# con merge por ["cod_mpio", "year"] — ver PASO 4.
# =============================================================

import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename=f"logs/03_gold_{datetime.today().strftime('%Y%m%d')}.log",
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)

SILVER_PATH = os.path.join("data", "silver")
GOLD_PATH   = os.path.join("data", "gold")
os.makedirs(GOLD_PATH, exist_ok=True)

# =============================================================
# PASO 1 — Cargar Silver DANE
# =============================================================

dane = pd.read_parquet(os.path.join(SILVER_PATH, "dane_silver.parquet"))
print(f"✅ Silver DANE: {dane.shape}")

# =============================================================
# PASO 2 — Panel base
# =============================================================

panel = dane.copy()

# =============================================================
# PASO 3 — Verificar balance
# =============================================================

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
# PASO 4 — Aquí se agregan las otras fuentes cuando las tengas
# Descomenta cada bloque cuando el silver correspondiente exista
# =============================================================

# --- SIMAT (cobertura secundaria) ---
# simat = pd.read_parquet(os.path.join(SILVER_PATH, "simat_silver.parquet"))
# panel = panel.merge(simat, on=["cod_mpio", "year"], how="left")

# --- DNP (desempeño fiscal) ---
# dnp = pd.read_parquet(os.path.join(SILVER_PATH, "dnp_silver.parquet"))
# panel = panel.merge(dnp, on=["cod_mpio", "year"], how="left")

# --- UARIV (desplazamiento forzado) ---
# uariv = pd.read_parquet(os.path.join(SILVER_PATH, "uariv_silver.parquet"))
# panel = panel.merge(uariv, on=["cod_mpio", "year"], how="left")

# --- INMLCF (homicidios) ---
# inmlcf = pd.read_parquet(os.path.join(SILVER_PATH, "inmlcf_silver.parquet"))
# panel  = panel.merge(inmlcf, on=["cod_mpio", "year"], how="left")

# =============================================================
# PASO 5 — Estadísticas descriptivas
# =============================================================

print("\n" + "="*55)
print("  ESTADÍSTICAS DESCRIPTIVAS — GOLD")
print("="*55)
print(panel[["va_total", "ln_va_total"]].describe().round(2).to_string())
print("="*55)

# =============================================================
# PASO 6 — Guardar Gold (3 formatos)
# =============================================================

panel.to_parquet(os.path.join(GOLD_PATH, "panel_gold.parquet"), index=False)
panel.to_csv(os.path.join(GOLD_PATH, "panel_gold.csv"), index=False, encoding="utf-8-sig")
panel.to_excel(os.path.join(GOLD_PATH, "panel_gold.xlsx"), index=False)

print(f"\n✅ Gold parquet : data/gold/panel_gold.parquet")
print(f"✅ Gold CSV     : data/gold/panel_gold.csv")
print(f"✅ Gold Excel   : data/gold/panel_gold.xlsx")

logging.info(f"Gold guardado: {panel.shape}")
print("\n✅ GOLD COMPLETADO — PANEL LISTO PARA ECONOMETRÍA")