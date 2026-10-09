import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pipeline import ejecutar_pipeline

app = FastAPI(
    title="Survey Processing API",
    version="1.0.0",
    description="API para exponer Signals, Moments y Cohort Intelligence"
)
df_completo, df_config, cohortes_data = ejecutar_pipeline()


def limpiar_para_json(obj):
    """
    Limpia recursivamente cualquier diccionario, lista o valor para convertir
    tipos de NumPy/Pandas (NaN, int64, float64, bool_) a tipos nativos de Python.
    """
    if isinstance(obj, dict):
        return {str(k): limpiar_para_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, set)):
        return [limpiar_para_json(v) for v in obj]
    elif pd.isna(obj):
        return None
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    return str(obj) if not isinstance(obj, (str, type(None))) else obj


@app.get("/")
def home():
    return {
        "status": "online",
        "message": "Survey Processing API está funcionando correctamente",
        "endpoints": ["/signals", "/moments", "/cohorts"]
    }


@app.get("/signals")
def get_signals():
    """Retorna las señales calculadas por participante."""
    senales_cols = sorted(df_config['signal'].unique().tolist())
    meta_cols = [c for c in df_completo.columns if c not in senales_cols and 'moment' not in c.lower()]
    
    df_resultado = df_completo[meta_cols + senales_cols]
    datos_limpios = limpiar_para_json(df_resultado.to_dict(orient="records"))
    
    return JSONResponse(content={"total": len(datos_limpios), "data": datos_limpios})


@app.get("/moments")
def get_moments():
    """Retorna los Career & Life Moments calculados por participante."""
    # Detecta cualquier columna que contenga 'moment' (sin importar mayúsculas/minúsculas)
    moments_cols = [c for c in df_completo.columns if 'moment' in c.lower()]
    senales_cols = df_config['signal'].unique().tolist()
    meta_cols = [c for c in df_completo.columns if c not in moments_cols and c not in senales_cols]
    
    df_resultado = df_completo[meta_cols + moments_cols]
    datos_limpios = limpiar_para_json(df_resultado.to_dict(orient="records"))
    
    return JSONResponse(content={"total": len(datos_limpios), "data": datos_limpios})


@app.get("/cohorts")
def get_cohorts():
    """Retorna el análisis completo de Cohort Intelligence."""
    datos_limpios = limpiar_para_json(cohortes_data)
    return JSONResponse(content={"data": datos_limpios})


@app.get("/cohorts/{dimension}/{value}")
def get_cohort_by_dimension_and_value(dimension: str, value: str):
    """Filtra las cohortes por dimensión y valor."""
    dimensiones_validas = ['age_group', 'role', 'work_mode']
    if dimension not in dimensiones_validas:
        raise HTTPException(
            status_code=400, 
            detail=f"Dimensión inválida '{dimension}'. Debe ser una de: {dimensiones_validas}"
        )
    
    filtrado = []
    for item in cohortes_data:
        cohorts_filtradas = [
            c for c in item["cohorts"] 
            if c["dimension"] == dimension and str(c["cohort"]).lower() == value.lower()
        ]
        if cohorts_filtradas:
            filtrado.append({
                "signal": item["signal"],
                "organization_score": item["organization_score"],
                "cohort": cohorts_filtradas[0]
            })
            
    if not filtrado:
        raise HTTPException(
            status_code=404, 
            detail=f"No se encontraron cohortes para {dimension}='{value}'"
        )
        
    datos_limpios = limpiar_para_json(filtrado)
    return JSONResponse(content={"dimension": dimension, "value": value, "data": datos_limpios})