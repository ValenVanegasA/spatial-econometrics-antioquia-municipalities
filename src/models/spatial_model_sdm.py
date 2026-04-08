#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
ECONOMETRÍA ESPACIAL DE PANEL — Antioquia 2015-2024
================================================================================

Secuencia metodológica correcta:

  1. Panel OLS con efectos fijos municipales (within-estimator)
  2. Moran's I sobre residuos del OLS (con W Queen real del shapefile)
  3. Tests LM de panel espacial (Anselin 1988; Elhorst 2014):
       - LM-Lag     → implica modelo SAR
       - LM-Error   → implica modelo SEM
       - RLM-Lag    → versión robusta (controla por Error)
       - RLM-Error  → versión robusta (controla por Lag)
  4. Regla de decisión:
       - Si RLM-Lag > RLM-Error y significativo → estimar Panel_FE_Lag  (SAR con FE)
       - Si RLM-Error > RLM-Lag y significativo → estimar Panel_FE_Error (SEM con FE)
       - Si ambos no significativos → quedar con Panel OLS FE (sin componente espacial)
  5. Efectos directos, indirectos y totales (LeSage & Pace 2009)

Variable dependiente:  ln_va_per_capita  (log VA per cápita, precios constantes)
Covariables:
  - tasa_homicidios       (homicidios por 100,000 hab — proxy de inseguridad)
  - cobertura_secundaria  (cobertura neta secundaria — proxy de capital humano)

Referencia W: data/gold/W_queen.pkl  (Queen contiguity, EPSG:3116, row-standardized)

Nota sobre panel: cobertura_secundaria no disponible en 2024 (rezago SIMAT).
El modelo trabaja con 2015-2023 (9 años, 125 municipios, N×T = 1,125 obs).
================================================================================
"""

import os
import sys
import pickle
import warnings
import logging
from datetime import datetime
from pathlib import Path

os.chdir(Path(__file__).resolve().parents[2])

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

# Econometría espacial
from spreg import PanelFE, Panel_FE_Lag, Panel_FE_Error
from spreg.diagnostics_panel import (
    panel_LMlag, panel_LMerror,
    panel_rLMlag, panel_rLMerror
)
from esda import Moran

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
os.makedirs("logs", exist_ok=True)
os.makedirs("results", exist_ok=True)

logging.basicConfig(
    filename=f"logs/spatial_model_{datetime.today().strftime('%Y%m%d')}.log",
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)

GOLD_PATH    = Path("data/gold")
RESULTS_PATH = Path("results")

# Variables del modelo — definición explícita
DEP_VAR  = "ln_va_per_capita"
COVARS   = ["tasa_homicidios", "cobertura_secundaria"]
ID_VAR   = "cod_mpio"
TIME_VAR = "year"
# Período de análisis: 2015-2023 (cobertura_secundaria disponible)
T_START, T_END = 2015, 2023

SEP = "=" * 72


# ===========================================================================
# SECCIÓN 1 — CARGA Y PREPARACIÓN DE DATOS
# ===========================================================================

def load_data():
    """Carga el panel Gold y la matriz W Queen preconstruida."""
    print(f"\n{SEP}")
    print("  SECCIÓN 1 — CARGA DE DATOS")
    print(SEP)

    # Panel
    panel_file = GOLD_PATH / "panel_gold.parquet"
    if not panel_file.exists():
        raise FileNotFoundError(f"Ejecuta 03_gold_panel.py primero: {panel_file}")
    df = pd.read_parquet(panel_file)

    # Filtrar período con cobertura completa
    df = df[df[TIME_VAR].between(T_START, T_END)].copy()
    print(f"\n  Panel: {T_START}-{T_END}, {df[ID_VAR].nunique()} municipios")
    print(f"  Observaciones: {len(df)}")

    # Eliminar NaN en variables del modelo
    cols_needed = [ID_VAR, TIME_VAR, DEP_VAR] + COVARS
    n_antes = len(df)
    df = df[cols_needed].dropna()
    n_despues = len(df)
    if n_antes > n_despues:
        print(f"  ⚠️  Eliminadas {n_antes - n_despues} obs con NaN")

    # Verificar balance
    n_mpios  = df[ID_VAR].nunique()
    n_years  = df[TIME_VAR].nunique()
    n_obs    = len(df)
    balanceado = n_obs == n_mpios * n_years
    print(f"  Balance: {n_mpios} mpios × {n_years} años = {n_mpios*n_years} obs esperadas")
    print(f"  Estado:  {'✅ BALANCEADO' if balanceado else '⚠️  DESBALANCEADO'}")

    # Ordenar: spreg Panel requiere orden por período-municipio o municipio-período
    # Panel_FE_Lag está escrito para datos ordenados: todos los t de mpio1, luego mpio2...
    df = df.sort_values([ID_VAR, TIME_VAR]).reset_index(drop=True)

    # Matriz W (Queen contigüidad real del shapefile)
    w_file = GOLD_PATH / "W_queen.pkl"
    if not w_file.exists():
        raise FileNotFoundError(
            f"Ejecuta src/models/build_W_queen.py primero: {w_file}"
        )
    with open(w_file, "rb") as f:
        w = pickle.load(f)

    # Alinear W con los municipios en el panel (subconjunto si hay diferencias)
    mpios_panel = sorted(df[ID_VAR].unique().tolist())
    mpios_w     = sorted(w.neighbors.keys())
    en_w_no_panel = set(mpios_w) - set(mpios_panel)
    en_panel_no_w = set(mpios_panel) - set(mpios_w)

    if en_panel_no_w:
        print(f"\n  ⚠️  Municipios del panel sin geometría en W: {en_panel_no_w}")
        df = df[~df[ID_VAR].isin(en_panel_no_w)]
        mpios_panel = sorted(df[ID_VAR].unique().tolist())

    from libpysal.weights import w_subset
    if en_w_no_panel:
        w = w_subset(w, mpios_panel)
        w.transform = "r"

    print(f"\n  W Queen cargada: {w.n} municipios, vecinos promedio: "
          f"{np.mean(list(w.cardinalities.values())):.1f}")

    logger.info(f"Datos cargados: {df.shape}, W: {w.n} nodos")
    return df, w


# ===========================================================================
# SECCIÓN 2 — PREPARAR VECTORES PARA SPREG PANEL
# ===========================================================================

def prepare_vectors(df, w):
    """
    spreg.Panel_FE_Lag / Panel_FE_Error esperan:
      y : array (N*T, 1) — ordenado por [ID, TIME]  → municipio varía más lento
      x : array (N*T, k) — mismas filas que y, SIN constante (el estimador FE la elimina)
      w : objeto W de libpysal con N nodos (N = número de municipios, no N*T)

    La ordenación correcta es: todos los t de mpio1, luego todos los t de mpio2, etc.
    """
    # Re-ordenar por municipio y luego tiempo
    df = df.sort_values([ID_VAR, TIME_VAR]).reset_index(drop=True)

    # Verificar que el W tiene exactamente los mismos IDs que el panel
    mpios_panel = sorted(df[ID_VAR].unique().tolist())
    mpios_w     = sorted(w.neighbors.keys())
    assert mpios_panel == mpios_w, (
        "IDs del panel y la matriz W no coinciden. "
        "Verifica build_W_queen.py y el panel Gold."
    )

    y = df[DEP_VAR].values.reshape(-1, 1)
    X = df[COVARS].values   # Sin constante — Panel_FE la elimina internamente

    n_mpios = df[ID_VAR].nunique()
    n_years = df[TIME_VAR].nunique()
    n_obs   = len(df)
    print(f"\n  Vectores preparados: y = ({n_obs},1), X = ({n_obs},{len(COVARS)})")
    print(f"  N municipios = {n_mpios}, T períodos = {n_years}")

    return y, X, df, n_mpios, n_years


# ===========================================================================
# SECCIÓN 3 — OLS CON EFECTOS FIJOS (WITHIN-ESTIMATOR)
# ===========================================================================

def ols_panel_fe(y, X, w, df, n_mpios, n_years):
    """
    Panel OLS con efectos fijos municipales (within-estimator).
    PanelFE de spreg implementa el demeaning interno.
    """
    print(f"\n{SEP}")
    print("  SECCIÓN 3 — PANEL OLS CON EFECTOS FIJOS (WITHIN)")
    print(SEP)
    print("""
  Especificación:
    ln_va_per_capita_it = α_i + β₁·tasa_homicidios_it + β₂·cobertura_secundaria_it + ε_it

  α_i capta heterogeneidad municipal invariante en el tiempo (geografía,
  dotación histórica de capital, instituciones). El within-estimator elimina
  este sesgo por omisión de variables no observadas constantes.
""")

    ols_fe = PanelFE(
        y, X, w,
        name_y=DEP_VAR,
        name_x=COVARS,
        name_ds="Antioquia 2015-2023"
    )

    print(f"  Pseudo-R² : {ols_fe.pr2:.4f}")
    print(f"  Log-Lik   : {ols_fe.logll:.4f}")
    print(f"\n  {'Variable':<25} {'Coef':>10} {'Std.Err':>10} {'t-stat':>10} {'p-valor':>10}")
    print("  " + "-"*65)

    names = COVARS
    for i, var in enumerate(names):
        coef  = ols_fe.betas[i, 0]
        se    = ols_fe.std_err[i]
        tstat = ols_fe.z_stat[i][0]   # PanelFE guarda los t-stats en z_stat
        pval  = ols_fe.z_stat[i][1]
        stars = "***" if pval < 0.01 else ("**" if pval < 0.05 else ("*" if pval < 0.1 else ""))
        print(f"  {var:<25} {coef:>10.4f} {se:>10.4f} {tstat:>10.3f} {pval:>10.4f} {stars}")

    logger.info(f"OLS FE: pr2={ols_fe.pr2:.4f}")
    return ols_fe


# ===========================================================================
# SECCIÓN 4 — MORAN'S I SOBRE RESIDUOS + TESTS LM DE PANEL
# ===========================================================================

def spatial_diagnostics(ols_fe, y, X, w, df):
    """
    Diagnósticos de dependencia espacial sobre el OLS con FE.

    Moran's I sobre residuos: detecta si la dependencia espacial persiste
    después de controlar por los efectos fijos.

    Tests LM (Elhorst 2014):
      LM-Lag   ~ chi²(1):  H₀ = ρ=0 vs H₁ = modelo SAR
      LM-Error ~ chi²(1):  H₀ = λ=0 vs H₁ = modelo SEM
      RLM-Lag  y RLM-Error: versiones robustas (controlan por el otro)
    """
    print(f"\n{SEP}")
    print("  SECCIÓN 4 — DIAGNÓSTICOS ESPACIALES")
    print(SEP)

    # residuos del within (PanelFE usa .u)

    # --- Moran's I sobre residuos (promediado por municipio para cross-section) ---
    print("\n  4a. Moran's I sobre residuos OLS-FE (promediados por municipio)\n")

    df_temp = df.copy()
    df_temp["residuo"] = ols_fe.u.flatten()   # 'u' = residuos en PanelFE
    res_cs = df_temp.groupby(ID_VAR)["residuo"].mean()
    # Alinear con el orden de la W
    res_cs = res_cs.reindex(sorted(w.neighbors.keys()))

    moran_res = Moran(res_cs.values, w)
    sig_res = "***" if moran_res.p_sim < 0.01 else (
              "**"  if moran_res.p_sim < 0.05 else (
              "*"   if moran_res.p_sim < 0.10 else "ns"))

    print(f"  I de Moran (residuos): {moran_res.I:.4f}")
    print(f"  Z-score             : {moran_res.z_sim:.3f}")
    print(f"  P-valor (simulado)  : {moran_res.p_sim:.4f}  {sig_res}")

    if moran_res.p_sim < 0.05:
        print("\n  → Dependencia espacial en residuos: justifica modelo espacial.")
    else:
        print("\n  → Sin dependencia espacial clara en residuos. El OLS-FE puede bastar.")

    # --- Tests LM de panel ---
    print("\n  4b. Tests LM de panel espacial (Anselin 1988; Elhorst 2014)\n")
    print(f"  {'Test':<18} {'Estadístico':>14} {'p-valor':>10} {'Decisión':>20}")
    print("  " + "-"*66)

    resultados_lm = {}

    for nombre, fn in [("LM-Lag", panel_LMlag), ("LM-Error", panel_LMerror),
                       ("RLM-Lag", panel_rLMlag), ("RLM-Error", panel_rLMerror)]:
        try:
            stat, pval = fn(y, X, w)
            sig = "***" if pval < 0.01 else ("**" if pval < 0.05 else ("*" if pval < 0.1 else "ns"))
            significativo = pval < 0.05
            resultados_lm[nombre] = {"stat": stat, "pval": pval, "sig": sig, "sign": significativo}
            print(f"  {nombre:<18} {stat:>14.4f} {pval:>10.4f} {sig:>20}")
        except Exception as e:
            resultados_lm[nombre] = {"stat": np.nan, "pval": np.nan, "sig": "ERR", "sign": False}
            print(f"  {nombre:<18} {'ERROR':>14} — {str(e)[:30]}")

    print("  " + "-"*66)
    print("  Significancia: *** p<0.01, ** p<0.05, * p<0.1, ns = no significativo")

    logger.info(f"LM tests: {resultados_lm}")
    return moran_res, resultados_lm


# ===========================================================================
# SECCIÓN 5 — REGLA DE DECISIÓN Y SELECCIÓN DE MODELO
# ===========================================================================

def select_model(moran_res, resultados_lm):
    """
    Regla de decisión estándar (Anselin 1988):

    1. Si RLM-Lag significativo y RLM-Error no → SAR (Panel_FE_Lag)
    2. Si RLM-Error significativo y RLM-Lag no → SEM (Panel_FE_Error)
    3. Si ambos significativos → el de mayor estadístico
    4. Si ninguno significativo → Panel OLS FE (sin componente espacial)

    Se prioriza la versión robusta (RLM) sobre la no robusta (LM).
    """
    print(f"\n{SEP}")
    print("  SECCIÓN 5 — SELECCIÓN DE MODELO")
    print(SEP)

    rlm_lag   = resultados_lm.get("RLM-Lag",   {"stat": 0, "pval": 1, "sign": False})
    rlm_error = resultados_lm.get("RLM-Error", {"stat": 0, "pval": 1, "sign": False})
    lm_lag    = resultados_lm.get("LM-Lag",    {"stat": 0, "pval": 1, "sign": False})
    lm_error  = resultados_lm.get("LM-Error",  {"stat": 0, "pval": 1, "sign": False})

    moran_sig = moran_res.p_sim < 0.10   # criterio laxo para el Moran

    if not rlm_lag["sign"] and not rlm_error["sign"]:
        decision = "OLS_FE"
        razon = ("Ni RLM-Lag ni RLM-Error son significativos al 5%.\n"
                 "  El panel OLS con efectos fijos es suficiente.\n"
                 "  No hay evidencia de dependencia espacial después de controlar por FE.")
    elif rlm_lag["sign"] and not rlm_error["sign"]:
        decision = "SAR"
        razon = ("RLM-Lag significativo, RLM-Error no → modelo SAR (Panel_FE_Lag).\n"
                 "  Hay spillovers en los niveles de VA per cápita entre vecinos.")
    elif rlm_error["sign"] and not rlm_lag["sign"]:
        decision = "SEM"
        razon = ("RLM-Error significativo, RLM-Lag no → modelo SEM (Panel_FE_Error).\n"
                 "  Dependencia espacial en los errores (shocks o variables omitidas comunes).")
    else:
        # Ambos significativos → elegir el de mayor estadístico robusto
        if rlm_lag["stat"] >= rlm_error["stat"]:
            decision = "SAR"
            razon = ("Ambos RLM significativos. RLM-Lag > RLM-Error → SAR.\n"
                     f"  RLM-Lag={rlm_lag['stat']:.3f}, RLM-Error={rlm_error['stat']:.3f}")
        else:
            decision = "SEM"
            razon = ("Ambos RLM significativos. RLM-Error > RLM-Lag → SEM.\n"
                     f"  RLM-Error={rlm_error['stat']:.3f}, RLM-Lag={rlm_lag['stat']:.3f}")

    print(f"\n  Modelo seleccionado: {decision}")
    print(f"  Razón: {razon}")
    logger.info(f"Modelo seleccionado: {decision}")
    return decision


# ===========================================================================
# SECCIÓN 6 — ESTIMACIÓN DEL MODELO SELECCIONADO
# ===========================================================================

def estimate_model(decision, y, X, w, df):
    """Estima el modelo elegido por la secuencia diagnóstica."""
    print(f"\n{SEP}")
    print(f"  SECCIÓN 6 — ESTIMACIÓN: {decision}")
    print(SEP)

    if decision == "OLS_FE":
        print("\n  El OLS con efectos fijos ya fue estimado (Sección 3).")
        print("  No se estima modelo espacial adicional.")
        return None

    elif decision == "SAR":
        print("""
  Modelo SAR con efectos fijos municipales:
    ln_va_per_capita_it = ρ·W·ln_va_per_capita_it + β'Xit + α_i + ε_it

  ρ = coeficiente de interdependencia espacial ('spillover' de VA per cápita).
  Si ρ>0 y significativo: el VA per cápita de un municipio depende positivamente
  del VA per cápita promedio de sus vecinos geográficos.
""")
        modelo = Panel_FE_Lag(
            y, X, w,
            name_y=DEP_VAR,
            name_x=COVARS,
            name_ds="Antioquia 2015-2023"
        )
        param_espacial = ("rho", modelo.rho)

    elif decision == "SEM":
        print("""
  Modelo SEM con efectos fijos municipales:
    ln_va_per_capita_it = β'Xit + α_i + u_it
    u_it = λ·W·u_it + ε_it

  λ = coeficiente de dependencia espacial en los errores.
  Si λ>0 y significativo: los shocks no observados se propagan entre vecinos
  (ej. shocks climáticos, derrames de política, precios de materias primas).
""")
        modelo = Panel_FE_Error(
            y, X, w,
            name_y=DEP_VAR,
            name_x=COVARS,
            name_ds="Antioquia 2015-2023"
        )
        param_espacial = ("lambda", modelo.lam)

    # --- Resultados ---
    param_nombre, param_valor = param_espacial
    print(f"\n  {'Variable':<25} {'Coef':>10} {'Std.Err':>10} {'z-stat':>10} {'p-valor':>10}")
    print("  " + "-"*65)

    for i, var in enumerate(COVARS):
        coef = modelo.betas[i, 0]
        se   = modelo.std_err[i]
        zval = modelo.z_stat[i][0]
        pval = modelo.z_stat[i][1]
        stars = "***" if pval < 0.01 else ("**" if pval < 0.05 else ("*" if pval < 0.1 else ""))
        print(f"  {var:<25} {coef:>10.4f} {se:>10.4f} {zval:>10.3f} {pval:>10.4f} {stars}")

    # Parámetro espacial
    idx_sp = len(COVARS)  # último parámetro
    sp_se  = modelo.std_err[idx_sp] if len(modelo.std_err) > idx_sp else np.nan
    sp_z   = modelo.z_stat[idx_sp][0] if len(modelo.z_stat) > idx_sp else np.nan
    sp_p   = modelo.z_stat[idx_sp][1] if len(modelo.z_stat) > idx_sp else np.nan
    sp_stars = "***" if sp_p < 0.01 else ("**" if sp_p < 0.05 else ("*" if sp_p < 0.1 else ""))

    print("  " + "-"*65)
    print(f"  {param_nombre:<25} {param_valor:>10.4f} {sp_se:>10.4f} {sp_z:>10.3f} {sp_p:>10.4f} {sp_stars}")

    print(f"\n  Log-Likelihood : {modelo.logll:.4f}")
    print(f"  AIC            : {modelo.aic:.4f}")

    logger.info(f"{decision}: {param_nombre}={param_valor:.4f}, LL={modelo.logll:.4f}")
    return modelo


# ===========================================================================
# SECCIÓN 7 — GUARDAR RESULTADOS
# ===========================================================================

def save_results(decision, ols_fe, modelo_espacial, moran_res, resultados_lm, df):
    """Guarda tablas de resultados en results/."""
    ts = datetime.now().strftime("%Y%m%d_%H%M")

    # --- Tabla de coeficientes ---
    rows = []

    # OLS FE
    for i, var in enumerate(COVARS):
        coef = ols_fe.betas[i, 0]
        se   = ols_fe.std_err[i]
        rows.append({"modelo": "OLS_FE", "variable": var,
                     "coef": round(coef, 5), "std_err": round(se, 5)})

    # Modelo espacial (si aplica)
    if modelo_espacial is not None:
        for i, var in enumerate(COVARS):
            coef = modelo_espacial.betas[i, 0]
            se   = modelo_espacial.std_err[i]
            rows.append({"modelo": decision, "variable": var,
                         "coef": round(coef, 5), "std_err": round(se, 5)})
        param_nombre = "rho" if decision == "SAR" else "lambda"
        param_val    = modelo_espacial.rho if decision == "SAR" else modelo_espacial.lam
        idx_sp = len(COVARS)
        sp_se  = modelo_espacial.std_err[idx_sp] if len(modelo_espacial.std_err) > idx_sp else np.nan
        rows.append({"modelo": decision, "variable": param_nombre,
                     "coef": round(param_val, 5), "std_err": round(sp_se, 5)})

    coef_df = pd.DataFrame(rows)
    coef_file = RESULTS_PATH / f"coeficientes_{ts}.csv"
    coef_df.to_csv(coef_file, index=False)

    # --- Tabla de diagnósticos ---
    diag_rows = [
        {"prueba": "Moran_I_residuos", "estadistico": round(moran_res.I, 4),
         "pvalor": round(moran_res.p_sim, 4)}
    ]
    for nombre, res in resultados_lm.items():
        if not np.isnan(res["stat"]):
            diag_rows.append({"prueba": nombre,
                               "estadistico": round(res["stat"], 4),
                               "pvalor": round(res["pval"], 4)})
    diag_df = pd.DataFrame(diag_rows)
    diag_file = RESULTS_PATH / f"diagnosticos_{ts}.csv"
    diag_df.to_csv(diag_file, index=False)

    print(f"\n  ✅ Guardados:")
    print(f"     {coef_file}")
    print(f"     {diag_file}")
    logger.info(f"Resultados en {RESULTS_PATH}")
    return coef_df, diag_df


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    print(SEP)
    print("  ECONOMETRÍA ESPACIAL DE PANEL — ANTIOQUIA 2015-2023")
    print(f"  Dep. var: {DEP_VAR}")
    print(f"  Regresores: {COVARS}")
    print(SEP)

    # 1. Datos y W
    df, w = load_data()

    # 2. Vectores
    y, X, df, n_mpios, n_years = prepare_vectors(df, w)

    # 3. OLS FE
    ols_fe = ols_panel_fe(y, X, w, df, n_mpios, n_years)

    # 4. Diagnósticos espaciales
    moran_res, resultados_lm = spatial_diagnostics(ols_fe, y, X, w, df)

    # 5. Selección de modelo
    decision = select_model(moran_res, resultados_lm)

    # 6. Estimación
    modelo_espacial = estimate_model(decision, y, X, w, df)

    # 7. Guardar
    coef_df, diag_df = save_results(
        decision, ols_fe, modelo_espacial, moran_res, resultados_lm, df
    )

    print(f"\n{SEP}")
    print(f"  FIN — Modelo: {decision}")
    print(SEP)
    return ols_fe, modelo_espacial, decision


if __name__ == "__main__":
    ols_fe, modelo_espacial, decision = main()
