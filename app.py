import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))

@st.cache_resource
def cargar_artefactos():
    perfil     = pd.read_csv("perfil_municipios.csv")
    km_final2  = joblib.load("modelo_kmeans_k3.pkl")
    scaler_km2 = joblib.load("scaler_kmeans_k3.pkl")
    perfil["score"] = perfil["tasa_accidentes"] * (1 + perfil["IDA_promedio"])
    return perfil, km_final2, scaler_km2

perfil, km_final2, scaler_km2 = cargar_artefactos()

etiquetas = {
    2: "🔴 Alto riesgo",
    1: "🟡 Riesgo medio",
    0: "🟢 Riesgo bajo"
}
perfil["nivel_riesgo"] = perfil["cluster"].map(etiquetas)

st.title("🚑 Distribución de Equipos de Emergencia")
st.subheader("Departamento de Antioquia — Municipios con registro histórico")

n_equipos = st.number_input(
    "Número total de equipos disponibles",
    min_value=1, max_value=500, value=100, step=1
)

if st.button("Calcular distribución"):
    n_equipos_original = int(n_equipos)
    n_equipos_calc = n_equipos_original
    n_municipios = len(perfil)
    df = perfil.copy()

    # Multiplicador por cluster
    tasa_por_cluster = df.groupby("cluster")["tasa_accidentes"].mean().rank(method="first")
    n_clusters = tasa_por_cluster.nunique()
    multiplicadores = {cluster: 0.8 + (rank - 1) * (0.4 / (n_clusters - 1))
                       for cluster, rank in tasa_por_cluster.items()}
    df["mult_cluster"] = df["cluster"].map(multiplicadores)
    df["score"] = df["tasa_accidentes"] * (1 + df["IDA_promedio"]) * df["mult_cluster"]

    df["equipos"] = 0

    if n_equipos_calc >= n_municipios * 3:
        df["equipos"] += 1
        n_equipos_calc -= n_municipios

    # Asignación proporcional con residuos
    proporcional = df["score"] / df["score"].sum() * n_equipos_calc
    df["equipos"] += proporcional.apply(np.floor).astype(int)
    df["residuo"] = proporcional - proporcional.apply(np.floor)

    # Ajuste final por residuos
    diferencia = n_equipos_original - df["equipos"].sum()
    if diferencia > 0:
        idx_sumar = df["residuo"].nlargest(diferencia).index
        df.loc[idx_sumar, "equipos"] += 1
    elif diferencia < 0:
        idx_quitar = df["residuo"].nsmallest(abs(diferencia)).index
        df.loc[idx_quitar, "equipos"] -= 1

    df = df.sort_values("equipos", ascending=False).reset_index(drop=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total equipos asignados", df["equipos"].sum())
    col2.metric("Municipios cubiertos",    (df["equipos"] > 0).sum())
    col3.metric("Mínimo por municipio",    df["equipos"].min())

    st.dataframe(
        df[["MUNICIPIO", "nivel_riesgo", "tasa_accidentes",
            "IDA_promedio", "poblacion", "equipos"]]
        .rename(columns={
            "MUNICIPIO":       "Municipio",
            "nivel_riesgo":    "Nivel de riesgo",
            "tasa_accidentes": "Tasa accidentes (x10k hab)",
            "IDA_promedio":    "IDA promedio",
            "poblacion":       "Población",
            "equipos":         "Equipos asignados"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Equipos por nivel de riesgo")
    resumen_cluster = (
        df.groupby("nivel_riesgo")["equipos"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "Total equipos", "count": "Municipios"})
        .reset_index()
        .rename(columns={"nivel_riesgo": "Nivel de riesgo"})
    )
    st.dataframe(resumen_cluster, use_container_width=True, hide_index=True)