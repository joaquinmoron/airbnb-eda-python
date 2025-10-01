from pathlib import Path
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns


# mostrar carpeta de ejecución (debug)
print("CWD:", os.getcwd())

# carpeta donde está este archivo .py
BASE_DIR = Path(__file__).resolve().parent

# leer usando la ruta del archivo .py
df_listings = pd.read_csv(BASE_DIR / "listings.csv", low_memory=False)

print("\nPrimeras 5 filas de listings:")
print(df_listings.head())

print("\nShape (filas, columnas):", df_listings.shape)

print("\nTipos de datos por columna (primeras 20):")
print(df_listings.dtypes.head(20))

print("\nInfo general:")
print(df_listings.info())

# Paso 3.1 — Nulos por columna (conteo y %)

total_rows = len(df_listings)
nulls = df_listings.isna().sum().sort_values(ascending=False)
nulls_pct = (nulls / total_rows * 100).round(2)

print("\nNulos por columna (Top 15):")
print(nulls.head(15))

print("\nPorcentaje de nulos por columna (Top 15):")
print(nulls_pct.head(15))

# Paso 3.2 — Filas duplicadas
dup_count = df_listings.duplicated().sum()
print(f"\nFilas duplicadas (idénticas): {dup_count}")

# Paso 3.3 — Cardinalidad (valores únicos)
nunique = df_listings.nunique(dropna=True).sort_values(ascending=False)
print("\nCardinalidad (valores únicos) — Top 15:")
print(nunique.head(15))

# Chequeo de key candidata
if "id" in df_listings.columns:
    print("\n¿listings.id es único?:", df_listings["id"].is_unique)


print("\n" + "="*70)
print("Cargando calendar.csv")

# 3.CAL.1 — Carga y vista rápida
df_calendar = pd.read_csv(BASE_DIR / "calendar.csv", low_memory=False)
print("\nPrimeras 5 filas de calendar:")
print(df_calendar.head())

print("\nShape (filas, columnas):", df_calendar.shape)
print("\nTipos de datos (primeras 20):")
print(df_calendar.dtypes.head(20))

print("\nInfo general:")
print(df_calendar.info())

# 3.CAL.2 — Nulos y duplicados
total_rows_cal = len(df_calendar)
nulls_cal = df_calendar.isna().sum().sort_values(ascending=False)
nulls_pct_cal = (nulls_cal / total_rows_cal * 100).round(2)

print("\nNulos por columna (Top 15):")
print(nulls_cal.head(15))

print("\nPorcentaje de nulos por columna (Top 15):")
print(nulls_pct_cal.head(15))

dup_count_cal = df_calendar.duplicated().sum()
print(f"\nFilas duplicadas (idénticas) en calendar: {dup_count_cal}")

# 3.CAL.3 — Cardinalidad y checks típicos
nunique_cal = df_calendar.nunique(dropna=True).sort_values(ascending=False)
print("\nCardinalidad (valores únicos) — Top 15 (calendar):")
print(nunique_cal.head(15))

# Claves/relaciones esperadas
if "listing_id" in df_calendar.columns:
    print("\n¿calendar.listing_id parece clave foránea hacia listings.id?:",
          "Sí, existe la columna listing_id" )
    print("Cantidad de listing_id únicos en calendar:", df_calendar["listing_id"].nunique())

# Fechas y disponibilidad (solo inspección, sin limpiar aún)
if "date" in df_calendar.columns:
    cal_dates = pd.to_datetime(df_calendar["date"], errors="coerce")
    print("\nRango de fechas en calendar:",
          cal_dates.min(), "→", cal_dates.max())
if "available" in df_calendar.columns:
    print("\nDistribución de 'available' (t/f/True/False/otros):")
    print(df_calendar["available"].astype(str).str.lower().value_counts(dropna=False).head(10))

print("\n" + "="*80)
print("Cargando reviews.csv")

# Paso 1: Cargar reviews.csv
df_reviews = pd.read_csv(BASE_DIR / "reviews.csv", low_memory=False)

# Paso 2: Primeras 5 filas
print("\nPrimeras 5 filas de reviews:")
print(df_reviews.head())

# Paso 3: Tamaño del dataset
print("\nShape (filas, columnas):", df_reviews.shape)

# Paso 4: Tipos de datos
print("\nTipos de datos (primeras 20 columnas):")
print(df_reviews.dtypes.head(20))

# Paso 5: Info general
print("\nInfo general:")
print(df_reviews.info())

# Paso 6: Nulos por columna (Top 15)
print("\nNulos por columna (Top 15):")
print(df_reviews.isnull().sum().sort_values(ascending=False).head(15))

# Paso 7: Porcentaje de nulos por columna (Top 15)
print("\nPorcentaje de nulos por columna (Top 15):")
print((df_reviews.isnull().sum() / len(df_reviews) * 100).sort_values(ascending=False).head(15))

# Paso 8: Filas duplicadas
print("\nFilas duplicadas (idénticas) en reviews:", df_reviews.duplicated().sum())

# Paso 9: Cardinalidad (valores únicos)
print("\nCardinalidad (valores únicos) - Top 15 columnas:")
print(df_reviews.nunique().sort_values(ascending=False).head(15))

# Paso 10: ¿review_id es único?
if "id" in df_reviews.columns:
    print("\n¿reviews.id es único?:", df_reviews["id"].is_unique)

# Paso 11: Rango de fechas si existe 'date'
if "date" in df_reviews.columns:
    try:
        df_reviews['date'] = pd.to_datetime(df_reviews['date'], errors='coerce')
    except (ValueError, TypeError) as e:
        print(f"No se pudo convertir la columna 'date' a datetime: {e}")


  


# ================================================================
# ETAPA 2 - LIMPIEZA PRELIMINAR
# DATASET: LISTINGS
# ================================================================

# ================================================================
# PASO 1 - Estandarizar nombres de columnas
# ================================================================
# Muchas columnas tienen espacios, mayúsculas o símbolos.
# Para evitar errores y mantener consistencia, convertimos los nombres
# a snake_case (minúsculas y guiones bajos).
# ================================================================

def standardize_columns(df):
    df = df.copy()
    df.columns = (
        df.columns
          .str.strip()
          .str.replace(r"[^\w\s]", "", regex=True)
          .str.replace(r"\s+", "_", regex=True)
          .str.lower()
    )
    return df

df_listings = standardize_columns(df_listings)
print("\nColumnas estandarizadas (primeras 20):")
print(df_listings.columns[:20])


# ================================================================
# PASO 2 - Eliminar columnas 100% nulas
# ================================================================
# Detectamos columnas que no tienen ningún valor (100% NaN).
# Estas columnas no aportan información, por lo que las eliminamos.
# ================================================================

nulls_total = df_listings.isna().sum()
cols_100_null = nulls_total[nulls_total == len(df_listings)].index.tolist()

print(f"\nColumnas con 100% nulos: ({len(cols_100_null)})")
print(cols_100_null)

df_listings.drop(columns=cols_100_null, inplace=True)
print(f"Shape tras eliminar 100% nulas: {df_listings.shape}")


# ================================================================
# PASO 3 - Detectar columnas con más del 80% de nulos
# ================================================================
# Listamos columnas con demasiados valores faltantes (>80%).
# No las eliminamos todavía, solo las marcamos para decidir más adelante
# según su relevancia en los análisis o dashboards.
# ================================================================

pct_nulls = (df_listings.isna().sum() / len(df_listings) * 100).sort_values(ascending=False)
high_null_cols = pct_nulls[pct_nulls > 80].index.tolist()

print(f"\nColumnas con >80% nulos: ({len(high_null_cols)})")
print(pct_nulls.loc[high_null_cols].round(2))


# ================================================================
# PASO 4 - Normalizar columnas de precios
# ================================================================
# Algunas columnas de precio vienen como texto con símbolos ($, ,).
# Convertimos todas las columnas que contengan "price" a numéricas (float).
# Esto es necesario para poder calcular KPIs, promedios, etc.
# ================================================================

def to_numeric_price(s):
    return pd.to_numeric(s.astype(str).str.replace(r"[^\d\.\-]", "", regex=True), errors="coerce")

price_like = [c for c in df_listings.columns if "price" in c]
print("\nColumnas tipo 'price' detectadas:", price_like)

for col in price_like:
    df_listings[col] = to_numeric_price(df_listings[col])

print(df_listings[price_like].describe().T.round(2))


# ================================================================
# PASO 5 - Convertir columnas categóricas
# ================================================================
# Algunas columnas representan categorías (ej. room_type, property_type).
# Las convertimos al tipo "category" para mejorar memoria y eficiencia
# en los análisis (agrupaciones, filtros, etc.).
# ================================================================

cat_candidates = [
    "room_type", "property_type", "neighbourhood_cleansed", "bed_type",
    "city", "state", "market"
]
present_cats = [c for c in cat_candidates if c in df_listings.columns]
for col in present_cats:
    df_listings[col] = df_listings[col].astype("category")

print("\nColumnas casteadas a category:", present_cats)


# ================================================================
# PASO 6 - Convertir columnas de IDs a enteros
# ================================================================
# Normalizamos columnas de IDs (id, host_id, etc.) a tipo Int64.
# Esto asegura que podamos hacer joins sin problemas y que las llaves
# sean consistentes.
# ================================================================

id_candidates = [c for c in df_listings.columns if c in ["id", "host_id"] or c.endswith("_id")]
for col in id_candidates:
    if df_listings[col].dtype == object:
        is_num_str = df_listings[col].astype(str).str.match(r"^\d+$", na=True)
        if is_num_str.any():
            df_listings[col] = pd.to_numeric(df_listings[col], errors="coerce").astype("Int64")
    elif pd.api.types.is_integer_dtype(df_listings[col]) or pd.api.types.is_float_dtype(df_listings[col]):
        df_listings[col] = pd.to_numeric(df_listings[col], errors="coerce").astype("Int64")

print("\nTipos de ID tras normalizar:")
print(df_listings[id_candidates].dtypes)


# ================================================================
# PASO 7 - Verificación de integridad básica
# ================================================================
# Validamos que la columna 'id' siga siendo única (clave primaria).
# También medimos uso de memoria para confirmar la optimización.
# ================================================================

print("\n¿listings.id sigue siendo único?:", df_listings["id"].is_unique if "id" in df_listings.columns else "no existe 'id'")
print("Uso de memoria (MB):", round(df_listings.memory_usage(deep=True).sum() / (1024**2), 2))


# ================================================================
# PASO 8 - Guardar dataset limpio
# ================================================================
# Exportamos la versión limpia de listings a una carpeta 'data/clean'.
# Esta versión servirá como base para SQL y Power BI.
# ================================================================

CLEAN_DIR = (BASE_DIR / "data" / "clean")
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

out_path = CLEAN_DIR / "listings_clean.csv"
df_listings.to_csv(out_path, index=False)
print(f"\nGuardado: {out_path}")

# ================================================================
# ETAPA 2 - LIMPIEZA PRELIMINAR
# DATASET: CALENDAR
# ================================================================

# (Plan B) Si df_calendar no existe por algún motivo, lo cargamos
try:
    df_calendar
except NameError:
    df_calendar = pd.read_csv(BASE_DIR / "calendar.csv", low_memory=False)
    print("df_calendar cargado desde disco (Plan B).")


# ================================================================
# PASO 0 - Vista rápida y duplicados (control de seguridad)
# ================================================================
# Confirmamos shape y chequeamos si existen filas duplicadas.
# Si hay duplicados, los eliminamos (esto evita sesgos en métricas).
# ================================================================

print("\n[CAL] Shape inicial:", df_calendar.shape)
dups_cal = df_calendar.duplicated().sum()
print("[CAL] Duplicados antes de limpiar:", dups_cal)
if dups_cal > 0:
    df_calendar = df_calendar.drop_duplicates()
    print("[CAL] Duplicados eliminados. Nuevo shape:", df_calendar.shape)
else:
    print("[CAL] No se encontraron duplicados.")


# ================================================================
# PASO 1 - Estandarizar nombres de columnas
# ================================================================
# Unificamos a snake_case para evitar errores al referenciar campos.
# Reutilizamos la misma función standardize_columns de Listings.
# ================================================================

df_calendar = standardize_columns(df_calendar)
print("\n[CAL] Columnas estandarizadas:", list(df_calendar.columns))


# ================================================================
# PASO 2 - Parsear fechas (date -> datetime)
# ================================================================
# Convertimos 'date' a tipo datetime para poder agrupar por día/mes/año.
# Guardamos el rango temporal para documentar la ventana de análisis.
# ================================================================

if "date" in df_calendar.columns:
    df_calendar["date"] = pd.to_datetime(df_calendar["date"], errors="coerce")
    print("[CAL] Rango de fechas:", df_calendar["date"].min(), "→", df_calendar["date"].max())
else:
    print("[CAL] No existe columna 'date'.")


# ================================================================
# PASO 3 - Normalizar 'available' a booleano
# ================================================================
# En crudo viene como 't'/'f' (o 'true'/'false'). Lo pasamos a True/False.
# Esto facilita filtros y cálculos de ocupación.
# ================================================================

if "available" in df_calendar.columns:
    m = df_calendar["available"].astype(str).str.lower().map({
        "t": True, "true": True, "f": False, "false": False
    })
    # si hay otros valores raros, los dejamos como NaN
    df_calendar["available"] = m
    print("[CAL] Distribución de 'available' (True/False/NaN):")
    print(df_calendar["available"].value_counts(dropna=False))
else:
    print("[CAL] No existe columna 'available'.")


# ================================================================
# PASO 4 - Convertir precios a numérico (price, adjusted_price)
# ================================================================
# 'price' y 'adjusted_price' vienen con símbolos ($ y comas).
# Los convertimos a float para KPIs como ADR, RevPAR, etc.
# ================================================================

price_cols_cal = [c for c in ["price", "adjusted_price"] if c in df_calendar.columns]
print("\n[CAL] Columnas de precio detectadas:", price_cols_cal)
for col in price_cols_cal:
    df_calendar[col] = to_numeric_price(df_calendar[col])

if price_cols_cal:
    print(df_calendar[price_cols_cal].describe().T.round(2))


# ================================================================
# PASO 5 - Tratar nulos puntuales en minimum_nights / maximum_nights
# ================================================================
# Detectamos que había ~105 nulos en ambas. Estrategias posibles:
#   - Rellenar con valores razonables (ej. mediana) para no perder filas.
#   - O eliminar solo esas filas si son muy pocas (impacto mínimo).
# Para este trabajo (documentado) imputamos con la MEDIANA del conjunto.
# ================================================================

for col in ["minimum_nights", "maximum_nights"]:
    if col in df_calendar.columns:
        n_before = df_calendar[col].isna().sum()
        if n_before > 0:
            med = df_calendar[col].median()
            df_calendar[col] = df_calendar[col].fillna(med)
            print(f"[CAL] Nulos imputados en {col}: {n_before} → imputado con mediana {med}.")
        else:
            print(f"[CAL] {col}: sin nulos.")
    else:
        print(f"[CAL] No existe columna '{col}'.")


# ================================================================
# PASO 6 - Tipar IDs a enteros con NaN permitido (Int64)
# ================================================================
# Aseguramos que 'listing_id' sea numérico estable para joins con listings.
# ================================================================

if "listing_id" in df_calendar.columns:
    if df_calendar["listing_id"].dtype == object:
        df_calendar["listing_id"] = pd.to_numeric(df_calendar["listing_id"], errors="coerce").astype("Int64")
    elif pd.api.types.is_integer_dtype(df_calendar["listing_id"]) or pd.api.types.is_float_dtype(df_calendar["listing_id"]):
        df_calendar["listing_id"] = pd.to_numeric(df_calendar["listing_id"], errors="coerce").astype("Int64")
    print("[CAL] dtype(listing_id):", df_calendar["listing_id"].dtype)
else:
    print("[CAL] No existe 'listing_id'.")


# ================================================================
# PASO 7 - Chequeos de integridad y sanity checks
# ================================================================
# - Verificamos que todos los listing_id de calendar existan en listings_clean.
# - Revisamos memoria.
# Nota: Para el match necesitamos df_listings (ya limpio) en memoria.
# ================================================================

if "listing_id" in df_calendar.columns and "id" in df_listings.columns:
    # set de ids de listings (PK)
    listings_ids = set(df_listings["id"].dropna().astype("Int64").tolist())
    cal_ids = set(df_calendar["listing_id"].dropna().astype("Int64").tolist())
    missing_in_listings = len(cal_ids - listings_ids)
    print(f"[CAL] listing_id distintos en calendar: {len(cal_ids)}")
    print(f"[CAL] listing_id que NO están en listings.id: {missing_in_listings}")
else:
    print("[CAL] No se pudo verificar integridad (faltan columnas).")

print("[CAL] Uso de memoria (MB):", round(df_calendar.memory_usage(deep=True).sum() / (1024**2), 2))


# ================================================================
# PASO 8 - Guardar dataset limpio
# ================================================================
# Exportamos 'calendar_clean.csv' en data/clean para usar en SQL / Power BI.
# ================================================================

CLEAN_DIR = (BASE_DIR / "data" / "clean")
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

out_path_cal = CLEAN_DIR / "calendar_clean.csv"
df_calendar.to_csv(out_path_cal, index=False)
print(f"\n[CAL] Guardado: {out_path_cal}")

# ================================================================
# ETAPA 2 - LIMPIEZA PRELIMINAR: REVIEWS
# ================================================================

print("\n===== REVIEWS - Limpieza Preliminar =====")

# Paso 1: Verificar duplicados
print("\nCantidad de duplicados antes de limpiar:", df_reviews.duplicated().sum())
df_reviews = df_reviews.drop_duplicates()
print("Cantidad de duplicados después de limpiar:", df_reviews.duplicated().sum())

# Paso 2: Calcular porcentaje de valores nulos por columna
print("\nPorcentaje de valores nulos por columna:")
print((df_reviews.isnull().mean() * 100).sort_values(ascending=False))

# Paso 3: Columnas con 100% de nulos (para descartar directo)
cols_all_null = df_reviews.columns[df_reviews.isnull().mean() == 1]
print("\nColumnas con 100% de nulos:", list(cols_all_null))
df_reviews = df_reviews.drop(columns=cols_all_null)

# Paso 4: Columnas con más del 80% de nulos (candidatas a descartar)
cols_high_null = df_reviews.columns[df_reviews.isnull().mean() > 0.8]
print("\nColumnas con más del 80% de nulos:", list(cols_high_null))

# Paso 5: Tipos de datos antes de transformar
print("\nTipos de datos antes de ajustar:")
print(df_reviews.dtypes)

# Paso 6: Identificar columnas categóricas y convertir a category
cat_cols = ["reviewer_id", "reviewer_name"]
for col in cat_cols:
    if col in df_reviews.columns:
        df_reviews[col] = df_reviews[col].astype("category")

# Paso 7: Identificar fechas y convertir a datetime
if "date" in df_reviews.columns:
    df_reviews["date"] = pd.to_datetime(df_reviews["date"], errors="coerce")

# Paso 8: Reporte de uso de memoria final
print("\nUso de memoria optimizado:")
print(df_reviews.info(memory_usage="deep"))

# Guardar dataset limpio preliminar

# Crear carpeta si no existe
os.makedirs("data/clean", exist_ok=True)

df_reviews.to_csv("data/clean/reviews_clean.csv", index=False)
print("\nArchivo guardado en: data/clean/reviews_clean.csv")

# ============================================================
# PASO 3 - ANÁLISIS DESCRIPTIVO (LISTINGS)
# ============================================================

# 3.1 - Estadísticas descriptivas generales
print("\n--- Estadísticas descriptivas generales de Listings ---")
print(df_listings['price'].describe())  # distribución de precios
print(df_listings['minimum_nights'].describe())  # noches mínimas
print(df_listings['number_of_reviews'].describe())  # reviews totales

# 3.2 - Percentiles de precios (para detectar outliers)
print("\n--- Percentiles de Precios ---")
print(df_listings['price'].quantile([0.25, 0.5, 0.75, 0.9, 0.95, 0.99]))

# 3.3 - Conteo de categorías principales
print("\n--- Distribución por tipo de habitación ---")
print(df_listings['room_type'].value_counts())

print("\n--- Distribución por tipo de propiedad ---")
print(df_listings['property_type'].value_counts().head(10))  # top 10

# 3.4 - Precio promedio por tipo de habitación
print("\n--- Precio promedio por tipo de habitación ---")
print(df_listings.groupby('room_type')['price'].mean())

# 3.5 - Precio promedio por barrio (Top 10 más caros)
print("\n--- Precio promedio por barrio (Top 10) ---")
print(df_listings.groupby('neighbourhood_cleansed')['price'].mean().sort_values(ascending=False).head(10))

# ================================================
# PASO 3 - ANÁLISIS DESCRIPTIVO (CALENDAR)
# ================================================

# 3.1 - Estadísticas descriptivas generales
print("\n--- Estadísticas descriptivas generales de Calendar ---")
print(df_calendar['price'].describe())  # distribución de precios

print(df_calendar['minimum_nights'].describe())  # noches mínimas
print(df_calendar['maximum_nights'].describe())  # noches máximas

# 3.2 - Percentiles de precios (para detectar outliers)
print("\n--- Percentiles de Precios ---")
print(df_calendar['price'].quantile([0.25, 0.5, 0.75, 0.9, 0.95, 0.99]))

# Filtrar outliers (<= percentil 99)
p99 = df_calendar['price'].quantile(0.99)
df_calendar_filtered = df_calendar[df_calendar['price'] <= p99]

print(f"\nSe filtraron precios mayores a {p99:.2f} (percentil 99)")
print(f"Shape original: {df_calendar.shape}, Shape filtrado: {df_calendar_filtered.shape}")

# 3.3 - Distribución de disponibilidad
print("\n--- Distribución de disponibilidad ---")
print(df_calendar['available'].value_counts(normalize=True) * 100)

# 3.4 - Precio promedio por mes (sin outliers)
df_calendar_filtered['month'] = df_calendar_filtered['date'].dt.to_period('M')
print("\n--- Precio promedio por mes (sin outliers) ---")
print(df_calendar_filtered.groupby('month')['price'].mean().head(12))  # primeros 12 meses

# ================================================
# PASO 3 - ANÁLISIS DESCRIPTIVO (REVIEWS)
# ================================================

# Nota: partimos de df_reviews ya limpio (date en datetime, ids ok)
# Para evitar SettingWithCopyWarning:
df_reviews = df_reviews.copy()

# 3.1 - Panorama general
print("\n--- Panorama general de Reviews ---")
total_reviews = len(df_reviews)
unique_listings = df_reviews["listing_id"].nunique() if "listing_id" in df_reviews.columns else None
unique_reviewers = df_reviews["reviewer_id"].nunique() if "reviewer_id" in df_reviews.columns else None
date_min = df_reviews["date"].min() if "date" in df_reviews.columns else None
date_max = df_reviews["date"].max() if "date" in df_reviews.columns else None

print(f"Total de reviews: {total_reviews:,}")
print(f"Listings con al menos 1 review: {unique_listings:,}")
print(f"Reviewers únicos: {unique_reviewers:,}")
print(f"Rango de fechas: {date_min} → {date_max}")

# 3.2 - Nulos en comentarios
print("\n--- Nulos en 'comments' ---")
if "comments" in df_reviews.columns:
    n_null = df_reviews["comments"].isna().sum()
    pct_null = n_null / len(df_reviews) * 100
    print(f"Nulos en comments: {n_null} ({pct_null:.4f}%)")
else:
    print("No existe la columna 'comments'.")

# 3.3 - Reviews por listing (distribución)
print("\n--- Reviews por listing (distribución) ---")
if "listing_id" in df_reviews.columns:
    reviews_por_listing = df_reviews["listing_id"].value_counts()
    print(reviews_por_listing.describe())             # stats
    print("\nTop 10 listings con más reviews:")
    print(reviews_por_listing.head(10))
else:
    print("No existe 'listing_id' para calcular distribución.")

# 3.4 - Evolución temporal de reviews (por mes y por año)
print("\n--- Evolución temporal de reviews ---")
if "date" in df_reviews.columns:
    df_reviews["month"] = df_reviews["date"].dt.to_period("M")
    df_reviews["year"]  = df_reviews["date"].dt.year

    reviews_por_mes = df_reviews.groupby("month")["id"].count() if "id" in df_reviews.columns else df_reviews.groupby("month").size()
    reviews_por_anio = df_reviews.groupby("year")["id"].count()  if "id" in df_reviews.columns else df_reviews.groupby("year").size()

    print("\nReviews por mes (primeros 12):")
    print(reviews_por_mes.head(12))
    print("\nReviews por año:")
    print(reviews_por_anio.sort_index())
else:
    print("No existe 'date' para análisis temporal.")

# 3.5 - Longitud de comentarios (calidad/cantidad de texto)
print("\n--- Longitud de comentarios (caracteres) ---")
if "comments" in df_reviews.columns:
    df_reviews["comment_len"] = df_reviews["comments"].astype(str).str.len()
    print(df_reviews["comment_len"].describe())
    print("\nPercentiles de longitud de comentarios:")
    print(df_reviews["comment_len"].quantile([0.25, 0.5, 0.75, 0.9, 0.95, 0.99]))
else:
    print("No existe la columna 'comments'.")

# 3.6 - Integridad con listings (match de IDs)
print("\n--- Integridad con Listings (match de IDs) ---")
if "listing_id" in df_reviews.columns and "id" in df_listings.columns:
    set_rev = set(df_reviews["listing_id"].dropna().astype("Int64"))
    set_lst = set(df_listings["id"].dropna().astype("Int64"))
    faltantes = len(set_rev - set_lst)
    print(f"Listings distintos en reviews: {len(set_rev):,}")
    print(f"Listings en reviews que NO están en listings: {faltantes:,}")
else:
    print("No se pudo verificar integridad (faltan columnas).")

# ================================================================
# VISUALIZACIONES EXPLORATORIAS – ETAPA 1 (EDA)
# ================================================================



sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)

# ------------------------------------------------
# Helper: carpeta de guardado (opcional)

os.makedirs("figs", exist_ok=True)

# ------------------------------------------------
# V1) DISTRIBUCIÓN DE PRECIOS (LISTINGS)
#   - Histograma (sin outliers y con outliers)
#   - Boxplot (sin outliers)
# ------------------------------------------------
print("\n[V1] Distribución de precios (listings)")
p99_price = df_listings["price"].quantile(0.99)
dfL_price = df_listings[df_listings["price"] <= p99_price].copy()

fig, ax = plt.subplots(1, 2, figsize=(14,5))
sns.histplot(df_listings["price"], bins=60, kde=True, ax=ax[0])
ax[0].set_title("Histograma de precios (con outliers)")
ax[0].set_xlabel("Precio (ARS)")

sns.histplot(dfL_price["price"], bins=60, kde=True, ax=ax[1], color="tab:green")
ax[1].set_title(f"Histograma de precios (<= P99 = {p99_price:.0f})")
ax[1].set_xlabel("Precio (ARS)")
plt.tight_layout()
plt.savefig("figs/V1_hist_precios_listings.png")
plt.show()

plt.figure(figsize=(8,4))
sns.boxplot(x=dfL_price["price"], color="tab:green")
plt.title("Boxplot de precios (listings) – sin outliers (<=P99)")
plt.xlabel("Precio (ARS)")
plt.savefig("figs/V1_box_precios_listings.png")
plt.show()

# ------------------------------------------------
# V2) NOCHES MÍNIMAS (LISTINGS)
#   - Histograma en escala log
#   - Boxplot (cap a percentil 99)
# ------------------------------------------------
print("\n[V2] Distribución de noches mínimas (listings)")
p99_nights = df_listings["minimum_nights"].quantile(0.99)
dfL_nights = df_listings[df_listings["minimum_nights"] <= p99_nights].copy()

plt.figure(figsize=(10,4))
sns.histplot(df_listings["minimum_nights"], bins=60)
plt.yscale("log")  # muchas barras bajas; log ayuda a ver la cola
plt.title("Histograma de minimum_nights (escala log)")
plt.xlabel("Noches mínimas")
plt.ylabel("Frecuencia (log)")
plt.savefig("figs/V2_hist_min_nights_log.png")
plt.show()

plt.figure(figsize=(8,4))
sns.boxplot(x=dfL_nights["minimum_nights"], color="tab:orange")
plt.title(f"Boxplot de minimum_nights (<=P99={p99_nights:.0f})")
plt.xlabel("Noches mínimas")
plt.savefig("figs/V2_box_min_nights.png")
plt.show()

# ------------------------------------------------
# V3) RELACIÓN PRECIO vs REVIEWS (LISTINGS)
#   - Scatter (con límite P99 de precio y reviews)
# ------------------------------------------------
print("\n[V3] Scatter: precio vs número de reseñas (listings)")
p99_price = df_listings["price"].quantile(0.99)
p99_rev   = df_listings["number_of_reviews"].quantile(0.99)
dfL_sc = df_listings[(df_listings["price"]<=p99_price) &
                     (df_listings["number_of_reviews"]<=p99_rev)].copy()

plt.figure(figsize=(8,6))
sns.scatterplot(data=dfL_sc, x="number_of_reviews", y="price", alpha=0.3)
plt.title("Precio vs #Reviews (<=P99 en ambas)")
plt.xlabel("Número de reseñas")
plt.ylabel("Precio (ARS)")
plt.savefig("figs/V3_scatter_price_reviews.png")
plt.show()

# ------------------------------------------------
# V4) PRECIO por TIPO DE HABITACIÓN (LISTINGS)
#   - Boxplot por categoría room_type (sin outliers en precio)
# ------------------------------------------------
print("\n[V4] Precio por room_type (listings)")
dfL_room = df_listings[df_listings["price"] <= df_listings["price"].quantile(0.99)].copy()
plt.figure(figsize=(8,5))
sns.boxplot(data=dfL_room, x="room_type", y="price")
plt.title("Precio por tipo de habitación (<=P99 precio)")
plt.xlabel("room_type")
plt.ylabel("Precio (ARS)")
plt.savefig("figs/V4_box_price_by_roomtype.png")
plt.show()

# ------------------------------------------------
# V5) DISPONIBILIDAD vs PRECIO (CALENDAR)
#   - Boxplot precio por available (t/f) – sin outliers
#   - Distribución % de available
# ------------------------------------------------
print("\n[V5] Precio vs Disponibilidad (calendar)")
p99_cal = df_calendar["price"].quantile(0.99)
dfC = df_calendar[df_calendar["price"] <= p99_cal].copy()

# Asegurar dtype booleano legible
dfC["available"] = dfC["available"].astype(str).replace({"t":"Disponible","f":"No disponible"})

plt.figure(figsize=(8,5))
sns.boxplot(data=dfC.sample(min(len(dfC), 200000), random_state=7), x="available", y="price")
plt.title("Precio por disponibilidad (calendar, muestra, <=P99)")
plt.xlabel("")
plt.ylabel("Precio (ARS)")
plt.savefig("figs/V5_box_price_available.png")
plt.show()

print("\nDistribución de disponibilidad (calendar):")
print(df_calendar["available"].value_counts(normalize=True)*100)

# ------------------------------------------------
# V6) EVOLUCIÓN TEMPORAL
#   a) Precios promedio por mes (calendar, sin outliers)
#   b) Reviews por mes (reviews)
# ------------------------------------------------
print("\n[V6a] Precio promedio por mes (calendar, sin outliers)")
dfC["month"] = dfC["date"].dt.to_period("M")
precio_mes = dfC.groupby("month")["price"].mean()

plt.figure(figsize=(10,4))
precio_mes.plot(marker="o")
plt.title("Precio promedio por mes (calendar, <=P99)")
plt.xlabel("Mes")
plt.ylabel("Precio (ARS)")
plt.grid(True, axis="y", alpha=0.3)
plt.savefig("figs/V6a_line_price_per_month.png")
plt.show()

print("\n[V6b] Reviews por mes (reviews)")
dfR = df_reviews.copy()
dfR["month"] = dfR["date"].dt.to_period("M")
rev_mes = dfR.groupby("month")["id"].count() if "id" in dfR.columns else dfR.groupby("month").size()

plt.figure(figsize=(10,4))
rev_mes.plot(marker="o", color="tab:purple")
plt.title("Cantidad de reviews por mes")
plt.xlabel("Mes")
plt.ylabel("# Reviews")
plt.grid(True, axis="y", alpha=0.3)
plt.savefig("figs/V6b_line_reviews_per_month.png")
plt.show()

# ------------------------------------------------
# V7) CORRELACIÓN (LISTINGS)
#   - Heatmap entre variables numéricas clave
# ------------------------------------------------
print("\n[V7] Matriz de correlación (listings)")
cols_num = ["price", "minimum_nights", "number_of_reviews", "availability_365"]
df_corr = df_listings[cols_num].copy()
corr = df_corr.corr(numeric_only=True)

plt.figure(figsize=(6,5))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
plt.title("Correlación (listings) – variables numéricas")
plt.savefig("figs/V7_heatmap_corr_listings.png")
plt.show()

# ================================================================
# FIN VISUALIZACIONES ETAPA 1
# Archivos PNG guardados en carpeta ./figs
# ================================================================
