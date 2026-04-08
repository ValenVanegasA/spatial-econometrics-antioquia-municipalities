#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
TUTORIAL: CONSTRUCCION DE MATRIZ W (QUEEN CONTIGUITY)
================================================================================

Este script es un tutorial completo para principiantes en econometria espacial.
Cada paso esta explicado en detalle.

OBJETIVO:
    Construir la matriz de pesos espaciales W usando contiguidad Queen
    para los 125 municipios de Antioquia.

CONCEPTOS CLAVE:
    1. W es una matriz N x N (N = numero de municipios)
    2. W[i,j] = peso de la conexion entre municipio i y municipio j
    3. Si i y j NO son vecinos: W[i,j] = 0
    4. Si i y j SI son vecinos: W[i,j] = 1 (o un peso calculado)
    5. La diagonal W[i,i] = 0 (un municipio no es vecino de si mismo)

ESTRUCTURA DE W:
              Mpio1  Mpio2  Mpio3  ...  MpioN
    Mpio1  [  0      1      0     ...    0   ]
    Mpio2  [  1      0      1     ...    0   ]
    Mpio3  [  0      1      0     ...    1   ]
    ...    [ ...    ...    ...   ...   ...  ]
    MpioN  [  0      0      1     ...    0   ]

TIPO: "Queen" significa que dos municipios son vecinos si comparten:
    - Un borde (frontera)  <-- como Rook
    - Un vertice (esquina)  <-- adicional a Rook

================================================================================
"""

import os
import sys
import json
import pickle
import logging
from pathlib import Path
from datetime import datetime

os.chdir(Path(__file__).resolve().parents[2])

import numpy as np
import pandas as pd
import geopandas as gpd
from libpysal.weights import Queen
from libpysal.weights import w_subset

# Configuracion de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Rutas
RAW_PATH = Path("data/raw")
GOLD_PATH = Path("data/gold")
GOLD_PATH.mkdir(exist_ok=True)

# Shapefile especifico
SHP_FOLDER = RAW_PATH / "shp"
SHP_FILE = SHP_FOLDER / "Municipios.shp"

print("="*80)
print("   TUTORIAL: CONSTRUYENDO LA MATRIZ W (MATRIZ DE PESOS ESPACIALES)")
print("="*80)

# =============================================================================
# PASO 1: CARGAR EL SHAPEFILE
# =============================================================================

print("\n" + "="*80)
print("PASO 1: CARGANDO EL SHAPEFILE (MAPA DE MUNICIPIOS)")
print("="*80)

print("""
Que es un Shapefile?
--------------------
Un shapefile es un formato de archivo que almacena datos geograficos.
Contiene:
  - La FORMA (geometria): los poligonos de cada municipio
  - Los ATRIBUTOS: codigo, nombre, area, etc.

Analogia: Es como un mapa de Google Maps pero en formato de datos
cientificos que Python puede leer.
""")

print(f"Archivo: {SHP_FILE}")
print(f"Existe: {SHP_FILE.exists()}")

if not SHP_FILE.exists():
    print(f"ERROR: No se encuentra el archivo {SHP_FILE}")
    sys.exit(1)

# Cargar con geopandas
gdf = gpd.read_file(SHP_FILE)

print(f"\n[+] Shapefile cargado exitosamente!")
print(f"    Total de municipios en Colombia: {len(gdf)}")
print(f"    Columnas: {list(gdf.columns)}")
print(f"    Sistema de coordenadas (CRS): {gdf.crs}")

# Ver las primeras filas
print(f"\n[+] Primeras 3 filas:")
print(gdf[['COD_MPIO', 'MPIO_NOMBR']].head(3).to_string(index=False))

# =============================================================================
# PASO 2: FILTRAR SOLO ANTIOQUIA (No necesario, el shp ya es de Antioquia)
# =============================================================================

print("\n" + "="*80)
print("PASO 2: VERIFICANDO MUNICIPIOS DE ANTIOQUIA")
print("="*80)

# El shapefile proporcionado ya contiene los 125 municipios de Antioquia
gdf_ant = gdf.copy()

print(f"[+] Municipios de Antioquia encontrados: {len(gdf_ant)}")
print(f"    (deberian ser 125)")

# Verificar que tenemos los 125
if len(gdf_ant) != 125:
    print(f"\n[!] ADVERTENCIA: Se esperaban 125 municipios, pero hay {len(gdf_ant)}")
    print("    Verificando codigos...")

# Mostrar ejemplos
print(f"\n[+] Ejemplos de municipios de Antioquia:")
sample = gdf_ant[['COD_MPIO', 'MPIO_NOMBR']].head(8)
for _, row in sample.iterrows():
    print(f"    {row['COD_MPIO']} - {row['MPIO_NOMBR']}")

# =============================================================================
# PASO 3: PREPARAR EL IDENTIFICADOR
# =============================================================================

print("\n" + "="*80)
print("PASO 3: PREPARANDO EL CODIGO DIVIPOLA COMO IDENTIFICADOR")
print("="*80)

print("""
Codigo DIVIPOLA
---------------
Es el codigo oficial de 5 digitos que identifica cada municipio:
- 05 = Departamento de Antioquia
- 001 = Municipio de Medellin
- 05001 = Codigo completo de Medellin

Por que es importante?
----------------------
La matriz W usa estos codigos para identificar a cada municipio.
Cuando hagamos el modelo, necesitamos que los codigos en W coincidan
EXACTAMENTE con los codigos en nuestro panel de datos.
""")

# Verificar que COD_MPIO tenga 5 digitos
gdf_ant['cod_mpio'] = gdf_ant['COD_MPIO'].astype(str).str.strip()

print(f"[+] Codigos DIVIPOLA preparados")
print(f"    Ejemplo de codigos:")
for i, row in gdf_ant[['cod_mpio', 'MPIO_NOMBR']].head(5).iterrows():
    print(f"      {row['cod_mpio']} -> {row['MPIO_NOMBR']}")

# =============================================================================
# PASO 4: CONSTRUIR LA MATRIZ W (QUEEN CONTIGUITY)
# =============================================================================

print("\n" + "="*80)
print("PASO 4: CONSTRUYENDO LA MATRIZ W (CONTIGUIDAD QUEEN)")
print("="*80)

print("""
Este es el corazon del proceso!
-------------------------------

Que hace Queen.from_dataframe()?
===============================

1. Toma cada poligono (municipio)
2. Verifica si toca a otros poligonos (frontera o esquina)
3. Si tocan: son "vecinos"
4. Crea una lista de vecinos para cada municipio

Ejemplo visual:
==============

    Municipio A [___] Municipio B
                |   |
                |___|
              Municipio C

En este ejemplo:
- A es vecino de B (comparten frontera superior)
- A es vecino de C (comparten vertice/esquina inferior)
- B es vecino de C (comparten frontera derecha)

En la matriz W:
    A  B  C
A [ 0  1  1 ]
B [ 1  0  1 ]
C [ 1  1  0 ]

Nota: La diagonal es 0 porque un municipio no es vecino de si mismo.
""")

print("[+] Construyendo matriz W (esto puede tomar unos segundos)...")

# Construir W usando Queen
# idVariable='cod_mpio' usa el codigo DIVIPOLA como identificador
w = Queen.from_dataframe(gdf_ant, idVariable='cod_mpio')

print("[+] Matriz W construida exitosamente!")

# =============================================================================
# PASO 5: ESTANDARIZAR LA MATRIZ W (ROW-STANDARDIZATION)
# =============================================================================

print("\n" + "="*80)
print("PASO 5: ESTANDARIZANDO LA MATRIZ W (ROW-STANDARDIZATION)")
print("="*80)

print("""
Por que estandarizamos?
-----------------------

La matriz W cruda tiene 1s y 0s:
- 1 = es vecino
- 0 = no es vecino

Pero si un municipio tiene 10 vecinos y otro tiene 2,
el primero tendria suma 10 y el segundo suma 2.

En econometria, queremos que el "efecto del vecindario" sea comparable.
Por eso dividimos cada fila entre el numero de vecinos.

Ejemplo:
========
Antes (crudo):
    A tiene 2 vecinos: [0, 1, 1, 0, 0] -> suma = 2
    B tiene 3 vecinos: [1, 0, 1, 1, 0] -> suma = 3

Despues (estandarizado):
    A: [0, 0.5, 0.5, 0, 0] -> suma = 1
    B: [0.333, 0, 0.333, 0.333, 0] -> suma = 1

Ahora ambos tienen el mismo "peso total" de vecinos.
Esto se llama "row-standardized" (estandarizado por filas).
""")

# Aplicar transformacion
w.transform = 'r'  # 'r' = row-standardized

print("[+] Matriz W estandarizada por filas (transform='r')")
print("    Cada fila suma 1.0")

# Verificar
sample_id = list(w.neighbors.keys())[0]
sample_weights = w.weights[sample_id]
sample_neighbors = w.neighbors[sample_id]

print(f"\n[+] Ejemplo de un municipio:")
print(f"    Municipio: {sample_id}")
print(f"    Numero de vecinos: {len(sample_neighbors)}")
print(f"    Pesos: {sample_weights[:5]}...")  # Primeros 5
print(f"    Suma de pesos: {sum(sample_weights):.4f} (debe ser 1.0)")

# =============================================================================
# PASO 6: ESTADISTICAS DESCRIPTIVAS DE W
# =============================================================================

print("\n" + "="*80)
print("PASO 6: ANALIZANDO LAS PROPIEDADES DE W")
print("="*80)

# Calcular estadisticas
n = w.n  # Numero de observaciones
cardinalities = list(w.cardinalities.values())  # Numero de vecinos por municipio

print(f"""
Resumen de la matriz W
======================

Numero de municipios (N): {n}
Conexiones totales: {w.s0:.0f}
Densidad de la matriz: {w.s0/(n*(n-1)):.3f} (proporcion de conexiones posibles)

Vecinos por municipio:
  Media: {np.mean(cardinalities):.1f}
  Mediana: {np.median(cardinalities):.1f}
  Minimo: {np.min(cardinalities)}
  Maximo: {np.max(cardinalities)}
  Desviacion estandar: {np.std(cardinalities):.2f}
""")

# Municipios con mas vecinos
max_neighbors = max(cardinalities)
min_neighbors = min(cardinalities)

print(f"[+] Municipio(s) con MAS vecinos ({max_neighbors}):")
for mpio, n_vecinos in w.cardinalities.items():
    if n_vecinos == max_neighbors:
        nombre = gdf_ant[gdf_ant['cod_mpio'] == mpio]['MPIO_NOMBR'].values
        if len(nombre) > 0:
            print(f"    {mpio} - {nombre[0]}")

print(f"\n[+] Municipio(s) con MENOS vecinos ({min_neighbors}):")
for mpio, n_vecinos in w.cardinalities.items():
    if n_vecinos == min_neighbors:
        nombre = gdf_ant[gdf_ant['cod_mpio'] == mpio]['MPIO_NOMBR'].values
        if len(nombre) > 0:
            print(f"    {mpio} - {nombre[0]}")

# Verificar si hay municipios sin vecinos (islas)
isolated = [m for m, c in w.cardinalities.items() if c == 0]
if isolated:
    print(f"\n[!] ADVERTENCIA: {len(isolated)} municipio(s) sin vecinos (islas):")
    for m in isolated:
        print(f"    {m}")
else:
    print(f"\n[+] Todos los municipios tienen al menos 1 vecino (bien!)")

# =============================================================================
# PASO 7: MOSTRAR EJEMPLOS DE VECINDARIOS
# =============================================================================

print("\n" + "="*80)
print("PASO 7: EJEMPLOS DE VECINDARIOS (PARA ENTENDER W)")
print("="*80)

print("""
Veamos ejemplos concretos de como se ven los vecinos en W.
""")

# Ejemplo 1: Medellin
cod_medellin = '05001'
if cod_medellin in w.neighbors:
    vecinos_med = w.neighbors[cod_medellin]
    print(f"[+] Ejemplo 1: MEDELLIN ({cod_medellin})")
    print(f"    Numero de vecinos: {len(vecinos_med)}")
    print(f"    Codigos de vecinos: {vecinos_med[:10]}")

    # Buscar nombres
    nombres_vecinos = []
    for v in vecinos_med[:10]:
        nombre = gdf_ant[gdf_ant['cod_mpio'] == v]['MPIO_NOMBR'].values
        if len(nombre) > 0:
            nombres_vecinos.append(nombre[0])
    print(f"    Nombres: {', '.join(nombres_vecinos[:5])}...")

# Ejemplo 2: Un municipio del oriente
cod_oriente = '05002'  # Abejorral
if cod_oriente in w.neighbors:
    vecinos_ab = w.neighbors[cod_oriente]
    print(f"\n[+] Ejemplo 2: ABEJORRAL ({cod_oriente})")
    print(f"    Numero de vecinos: {len(vecinos_ab)}")
    nombres_ab = []
    for v in vecinos_ab[:8]:
        nombre = gdf_ant[gdf_ant['cod_mpio'] == v]['MPIO_NOMBR'].values
        if len(nombre) > 0:
            nombres_ab.append(nombre[0])
    print(f"    Vecinos: {', '.join(nombres_ab)}")

# =============================================================================
# PASO 8: VALIDAR CON EL PANEL DE DATOS
# =============================================================================

print("\n" + "="*80)
print("PASO 8: VALIDANDO CONTRA EL PANEL GOLD")
print("="*80)

print("""
Por que validamos?
------------------
Necesitamos que los municipios en W sean EXACTAMENTE los mismos
que los municipios en nuestro panel de datos.

Si hay diferencias, el modelo no funcionara.
""")

# Cargar panel
panel = pd.read_parquet(GOLD_PATH / "panel_gold.parquet")
panel_municipios = set([str(x).zfill(5) for x in panel['cod_mpio'].unique()])
w_municipios = set(w.neighbors.keys())

print(f"[+] Municipios en panel_gold: {len(panel_municipios)}")
print(f"[+] Municipios en W: {len(w_municipios)}")

# Verificar coincidencia
faltan_en_w = panel_municipios - w_municipios
sobran_en_w = w_municipios - panel_municipios

if faltan_en_w:
    print(f"\n[!] Municipios en panel pero NO en W: {len(faltan_en_w)}")
    for m in list(faltan_en_w)[:5]:
        print(f"    - {m}")
else:
    print(f"\n[+] Todos los municipios del panel estan en W (perfecto!)")

if sobran_en_w:
    print(f"\n[!] Municipios en W pero NO en panel: {len(sobran_en_w)}")
    print(f"    (Esto esta bien, solo usaremos los que estan en ambos)")

# =============================================================================
# PASO 9: GUARDAR W EN DIFERENTES FORMATOS
# =============================================================================

print("\n" + "="*80)
print("PASO 9: GUARDANDO W EN DIFERENTES FORMATOS")
print("="*80)

print("""
Guardamos W en 3 formatos para diferentes usos:

1. .pkl (Pickle):
   - Formato nativo de Python
   - Se carga directamente con: pickle.load()
   - Mantiene todos los metodos de W

2. .gal (GeoDa format):
   - Formato estandar de GeoDa (software de econometria espacial)
   - Interoperable con R, Stata, GeoDa
   - Es un archivo de texto legible

3. .json (Metadatos):
   - Informacion resumida sobre W
   - Para documentacion y referencia rapida
""")

# 1. Guardar como Pickle (formato Python)
pkl_path = GOLD_PATH / "W_queen.pkl"
with open(pkl_path, 'wb') as f:
    pickle.dump(w, f)
print(f"[+] Guardado: {pkl_path}")

# 2. Guardar como GAL (formato GeoDa)
gal_path = GOLD_PATH / "W_queen.gal"
try:
    w.to_file(str(gal_path))
    print(f"[+] Guardado: {gal_path}")
except Exception as e:
    print(f"[!] Error guardando GAL: {e}")
    # Alternativa: crear archivo manualmente
    with open(gal_path, 'w') as f:
        f.write(f"0 {w.n} Queen contiguity\n")
        for mpio in w.neighbors:
            vecinos = w.neighbors[mpio]
            f.write(f"{mpio} {len(vecinos)}\n")
            if vecinos:
                f.write(" ".join(vecinos) + "\n")
    print(f"[+] Guardado (manual): {gal_path}")

# 3. Guardar metadatos JSON
json_path = GOLD_PATH / "W_queen_info.json"
metadata = {
    "created": datetime.now().isoformat(),
    "method": "Queen contiguity",
    "transformation": "row-standardized (r)",
    "n_municipios": n,
    "n_connections": int(w.s0),
    "density": w.s0/(n*(n-1)),
    "neighbors_stats": {
        "mean": float(np.mean(cardinalities)),
        "median": float(np.median(cardinalities)),
        "min": int(np.min(cardinalities)),
        "max": int(np.max(cardinalities)),
        "std": float(np.std(cardinalities))
    },
    "id_column": "cod_mpio",
    "shapefile": "Munpio.shp",
    "validation": {
        "panel_municipios": len(panel_municipios),
        "w_municipios": len(w_municipios),
        "match": len(panel_municipios & w_municipios)
    },
    "municipios_list": sorted(list(w.neighbors.keys()))
}

with open(json_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"[+] Guardado: {json_path}")

# =============================================================================
# PASO 10: RESUMEN FINAL
# =============================================================================

print("\n" + "="*80)
print("PASO 10: RESUMEN FINAL Y PROXIMOS PASOS")
print("="*80)

print(f"""
MATRIZ W CONSTRUIDA EXITOSAMENTE
================================

Estadisticas:
- Municipios: {n}
- Conexiones totales: {w.s0:.0f}
- Vecinos promedio: {np.mean(cardinalities):.1f}
- Metodo: Queen contiguity
- Estandarizacion: Row-standardized

Archivos guardados en data/gold/:
  1. W_queen.pkl      -> Para usar en Python
  2. W_queen.gal      -> Para usar en GeoDa/Stata/R
  3. W_queen_info.json -> Metadatos y documentacion

QUE HACE W EN EL MODELO?
=========================

En el modelo SAR (Spatial Autoregressive):

    y = rho * W*y + X*beta + epsilon
        ^^^^^^^^^^
        Esta parte es la clave!

W*y significa: "El promedio ponderado de y en los municipios vecinos"

Ejemplo para Medellin:
    Wy[Medellin] = (1/n_vecinos) * sum(y[vecinos de Medellin])

Si rho es significativo y positivo, significa que:
"El valor agregado de Medellin depende del valor agregado promedio
de sus municipios vecinos"

Esto captura el efecto de "spillover" o "derrame espacial".

PROXIMO PASO:
=============
Ahora que tenemos W, podemos:
1. Recalcular el Moran's I con W real (mas preciso)
2. Estimar el modelo SAR con W real
3. Calcular efectos directos e indirectos

Estas listo para continuar?
""")

print("="*80)
print("   FIN DEL TUTORIAL - W CONSTRUIDA EXITOSAMENTE")
print("="*80)

# Guardar tambien un resumen CSV para facil lectura
summary_data = []
for mpio in w.neighbors:
    n_vecinos = len(w.neighbors[mpio])
    nombre = gdf_ant[gdf_ant['cod_mpio'] == mpio]['MPIO_NOMBR'].values
    nombre = nombre[0] if len(nombre) > 0 else 'NA'
    subregion = panel[panel['cod_mpio'] == mpio]['subregion'].unique()
    subregion = subregion[0] if len(subregion) > 0 else 'NA'
    summary_data.append({
        'cod_mpio': mpio,
        'municipio': nombre,
        'subregion': subregion,
        'n_vecinos': n_vecinos,
        'vecinos_codes': ';'.join(w.neighbors[mpio][:5]) + ('...' if len(w.neighbors[mpio]) > 5 else '')
    })

summary_df = pd.DataFrame(summary_data)
summary_df = summary_df.sort_values('cod_mpio')
summary_df.to_csv(GOLD_PATH / "W_neighbors_summary.csv", index=False)
print(f"\n[+] Guardado resumen: {GOLD_PATH / 'W_neighbors_summary.csv'}")

print("\n[+] Proceso completado!")
