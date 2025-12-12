from fastapi import FastAPI
from typing import List, Dict
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import ast

app = FastAPI(
    title = "Mini API - Análisis ",
    description = "API local para exponer análisis descriptivos de películas.",
    version = "1.0.0"
)

# =========================
# CARGA Y PREPROCESADO
# =========================

def parse_list_column(s):
    """Convierte strings tipo lista (JSON) en listas de Python."""
    try:
        return ast.literal_eval(s) if isinstance(s, str) else []
    except Exception:
        return []

def _clean_nans(obj):
    """Reemplaza NaN/inf por None de forma recursiva (para JSON válido)."""
    if isinstance(obj, dict):
        return {k: _clean_nans(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_nans(v) for v in obj]
    try:
        if isinstance(obj, (float, np.floating)) and (np.isnan(obj) or np.isinf(obj)):
            return None
    except Exception:
        pass
    return obj

def as_json(payload):
    """Devuelve respuesta JSON garantizando serialización correcta."""
    payload = _clean_nans(payload)
    return JSONResponse(content=jsonable_encoder(payload))

# Cargar datasets
movies = pd.read_csv("tmdb_5000_movies_limpia.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")

# Parseo de columnas JSON-like
movies["genres_parsed"] = movies["genres"].apply(parse_list_column)
movies["countries_parsed"] = movies["production_countries"].apply(parse_list_column)
credits["crew_parsed"] = credits["crew"].apply(parse_list_column)

# =========================
# 1) ROI POR GÉNERO Y PAÍS
# =========================

# ROI = revenue / budget
movies["ROI"] = np.where(
    movies["budget"] > 0,
    movies["revenue"] / movies["budget"],
    np.nan
)

movies_roi = movies.dropna(subset=["ROI"]).copy()

# --- ROI por género ---
rows_genre = []
for _, row in movies_roi.iterrows():
    roi = row["ROI"]
    for g in row["genres_parsed"]:
        rows_genre.append({"genre": g.get("name"), "ROI": roi})

genre_roi = pd.DataFrame(rows_genre)

genre_stats = (
    genre_roi
    .groupby("genre")["ROI"]
    .agg(count="count", roi_mean="mean", roi_median="median")
    .sort_values("roi_median", ascending=False)
).reset_index()

# --- ROI por país ---
rows_country = []
for _, row in movies_roi.iterrows():
    roi = row["ROI"]
    for c in row["countries_parsed"]:
        rows_country.append({"country": c.get("name"), "ROI": roi})

country_roi = pd.DataFrame(rows_country)

country_stats = (
    country_roi
    .groupby("country")["ROI"]
    .agg(count="count", roi_mean="mean", roi_median="median")
    .sort_values("roi_median", ascending=False)
).reset_index()

# =================================
# 2) CORRELACIÓN PRESUPUESTO-RATING
# =================================

df_br = movies[movies["budget"] > 0].copy()

pearson = df_br["budget"].corr(df_br["vote_average"], method="pearson")
spearman = df_br["budget"].corr(df_br["vote_average"], method="spearman")

df_br["log_budget"] = np.log10(df_br["budget"])
pearson_log = df_br["log_budget"].corr(df_br["vote_average"], method="pearson")
spearman_log = df_br["log_budget"].corr(df_br["vote_average"], method="spearman")

correlaciones_presupuesto_rating = {
    "pearson_presupuesto_rating": float(pearson),
    "spearman_presupuesto_rating": float(spearman),
    "pearson_log_presupuesto_rating": float(pearson_log),
    "spearman_log_presupuesto_rating": float(spearman_log),
}

# =========================
# 3) DIRECTORES Y RATINGS
# =========================

rows_directors = []
for _, row in credits.iterrows():
    movie_id = row["movie_id"]
    for person in row["crew_parsed"]:
        if person.get("job") == "Director":
            rows_directors.append({
                "movie_id": movie_id,
                "director": person.get("name")
            })

directors_df = pd.DataFrame(rows_directors)

directors_movies = directors_df.merge(
    movies[["id", "title", "vote_average"]],
    left_on="movie_id",
    right_on="id",
    how="left"
)

dir_stats = (
    directors_movies
    .groupby("director")["vote_average"]
    .agg(count="count", rating_mean="mean")
    .sort_values("rating_mean", ascending=False)
).reset_index()

# =========================
# ENDPOINTS
# =========================

@app.get("/")
def root():
    return {
        "mensaje": "Mini API TMDB funcionando",
        "endpoints": [
            "/roi_por_genero",
            "/roi_por_pais",
            "/top_directores",
            "/correlacion_presupuesto_rating",
        ]
    }

@app.get("/roi_por_genero")
def roi_por_genero(top_n: int = 10):
    """
    Devuelve los géneros con mayor mediana de ROI.
    Parámetro: top_n (por defecto 10).
    """
    df = genre_stats.head(top_n)
    return as_json(df.to_dict(orient="records"))


@app.get("/roi_por_pais")
def roi_por_pais(min_peliculas: int = 30, top_n: int = 10):
    """
    Devuelve países con mayor mediana de ROI.
    - min_peliculas: mínimo de películas para considerar el país.
    - top_n: cuántos países devolver (ordenados por mediana de ROI).
    """
    df = (
        country_stats[country_stats["count"] >= min_peliculas]
        .head(top_n)
    )
    return as_json(df.to_dict(orient="records"))


@app.get("/top_directores")
def top_directores(min_peliculas: int = 4, top_n: int = 10):
    """
    Devuelve directores con mejor rating promedio.
    - min_peliculas: mínimo de películas dirigidas.
    - top_n: cuántos directores devolver.
    """
    df = (
        dir_stats[dir_stats["count"] >= min_peliculas]
        .head(top_n)
    )
    return as_json(df.to_dict(orient="records"))


@app.get("/correlacion_presupuesto_rating")
def correlacion_presupuesto_rating():
    """
    Devuelve las correlaciones entre presupuesto y rating.
    Incluye presupuesto lineal y log10(presupuesto).
    """
    return as_json(correlaciones_presupuesto_rating)