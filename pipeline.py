import numpy as np
import pandas as pd
ruta_respuestas = 'responses.csv'
ruta_config = 'item_config.csv'
def cargar_datos(ruta_respuestas: str, ruta_config: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_resp = pd.read_csv(ruta_respuestas)
    df_config = pd.read_csv(ruta_config)
    return df_resp, df_config
def validar_respuestas(df_resp: pd.DataFrame, df_config: pd.DataFrame) -> pd.DataFrame:
    df = df_resp.drop_duplicates().copy()
    items = df_config['item_code'].unique().tolist()
    for col in items:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        fuera_de_rango = df[col].notna() & (~df[col].between(1, 5))
        df.loc[fuera_de_rango, col] = np.nan
    return df

def transformar_respuestas(df_resp: pd.DataFrame, df_config: pd.DataFrame) -> pd.DataFrame:
    df_trans = df_resp.copy()
    escala = {1: 0, 2: 25, 3: 50, 4: 75, 5: 100}
    for _, fila in df_config.iterrows():
        item = fila['item_code']
        val = df_trans[item]
        if bool(fila['reverse_flag']):
            val = 6 - val
        df_trans[item] = val.map(escala)
    return df_trans

def calcular_senales(df_resp: pd.DataFrame, df_trans: pd.DataFrame, df_config: pd.DataFrame) -> pd.DataFrame:
    items = df_config['item_code'].tolist()
    meta = [c for c in df_resp.columns if c not in items]
    df_res = df_resp[meta].copy()
    for senal, grupo in df_config.groupby('signal'):
        items_s = grupo['item_code'].tolist()
        pesos = grupo['weight'].values
        vals = df_trans[items_s]
        suma_valores = vals.mul(pesos, axis=1).sum(axis=1, skipna=True)
        suma_pesos = vals.notna().mul(pesos, axis=1).sum(axis=1)
        df_res[senal] = suma_valores / suma_pesos
    return df_res

def inferir_moments(df_resultado: pd.DataFrame) -> pd.DataFrame:
    df_moments = df_resultado.copy()
    df_moments['moment_learning_exploration'] = (df_resultado['Learning Appetite'] * 0.6+ df_resultado['Learning Capacity'] * 0.4)
    df_moments['moment_mastery_contribution'] = (df_resultado['Professional Agency'] * 0.5+ df_resultado['Knowledge Exchange'] * 0.5)
    df_moments['moment_consolidation_stability'] = (df_resultado['Future Visibility'] * 0.4+ df_resultado['Development Access'] * 0.4+ (100 - df_resultado['Learning Appetite']) * 0.2)
    df_moments['moment_transition_reinvention'] = (df_resultado['Learning Appetite'] * 0.4+ df_resultado['Professional Agency'] * 0.4+ (100 - df_resultado['Future Visibility']) * 0.2)
    return df_moments

def analizar_cohortes(df_resultado: pd.DataFrame, df_config: pd.DataFrame) -> list:
    variables_cohorte = ['age_group', 'role', 'work_mode']
    senales = sorted(df_config['signal'].unique().tolist())
    reporte_cohortes = []

    for senal in senales:
        org_score = float(round(df_resultado[senal].mean(), 1))
        lista_cohortes = []

        for var in variables_cohorte:
            if var not in df_resultado.columns:
                continue
            for valor_grupo, df_grupo in df_resultado.groupby(var):
                n = int(df_grupo[senal].notna().sum())
                reportable = bool(n >= 10)
                
                if reportable:
                    score = float(round(df_grupo[senal].mean(), 1))
                    delta = float(round(score - org_score, 1))
                else:
                    score = None
                    delta = None

                lista_cohortes.append({
                    "dimension": var,
                    "cohort": str(valor_grupo),
                    "n": n,
                    "score": score,
                    "delta": delta,
                    "reportable": reportable
                })

        reporte_cohortes.append({
            "signal": senal,
            "organization_score": org_score,
            "cohorts": lista_cohortes
        })

    return reporte_cohortes
def ejecutar_pipeline(archivo_resp: str = ruta_respuestas, archivo_cfg: str = ruta_config):
    df_resp, df_config = cargar_datos(archivo_resp, archivo_cfg)
    df_val = validar_respuestas(df_resp, df_config)
    df_trans = transformar_respuestas(df_val, df_config)
    df_senales = calcular_senales(df_val, df_trans, df_config)
    df_completo = inferir_moments(df_senales)
    cohortes = analizar_cohortes(df_completo, df_config)
    return df_completo, df_config, cohortes
pd.set_option('display.max_columns', None)
if __name__ == "__main__":
    df_completo, df_config, cohortes = ejecutar_pipeline()
    print("Procesamiento completado con éxito.")
    print("\n=== RESULTADO INDIVIDUAL ===")
    print(df_completo)
    print("\n=== COHORT INTELLIGENCE ===")

    reporte_cohortes = analizar_cohortes(
        df_completo,
        df_config
    )
    for bloque in reporte_cohortes:
        print(f"\nSignal: {bloque['signal']}")
        print(
            f"Organization score: "
            f"{bloque['organization_score']}"
        )
        for cohorte in bloque["cohorts"]:
            print(cohorte)