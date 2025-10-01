# Scripts Python – Etapa 1 (EDA + Limpieza preliminar)

Este directorio contiene el trabajo en **Python** de la primera etapa del proyecto Digital House (EDA y limpieza de los 3 datasets originales de Airbnb).

## Contenido
- **Etapa1DH.py**  
  Script principal en Python. Realiza la exploración inicial (EDA) y la limpieza preliminar de:
  - `listings.csv`
  - `calendar.csv`
  - `reviews.csv`

- **/data/clean/**  
  Carpeta con los datasets ya limpios generados por el script:
  - `listings_clean.csv`
  - `calendar_clean.csv`
  - `reviews_clean.csv`

## Proceso
1. Se cargan los datasets originales de Airbnb (entregados por Digital House).
2. Se realiza limpieza y estandarización (nulos, duplicados, tipos de datos, outliers básicos).
3. Se generan los datasets limpios en `/data/clean/`.

## Notas
- Estos CSV son los resultados **intermedios de Python**.  
- La segunda etapa (SQL) tomó estos archivos como insumo para generar las tablas finales (`dim_listing`, `dim_date`, `fact_calendar`, `reviews_unmatched`) que se usaron en Power BI.  
- Los datasets **finales** se encuentran en `/01_datos_limpios/`.
