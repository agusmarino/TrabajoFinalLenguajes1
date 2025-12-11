from fastapi import FastAPI
from typing import List, Dict
import pandas as pd
import numpy as np
import ast

app = FastAPI(
    title = "Mini API - Análisis TMDB",
    description = "API local para exponer análisis descriptivos de películas (TMDB).",
    version = "1.0.0"
)

