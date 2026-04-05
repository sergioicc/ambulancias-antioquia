import streamlit as st
import pandas as pd
import joblib
import os
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
    0: "🔴 Alto riesgo",
    1: "🟡 Riesgo medio",
    2: "🟢 Riesgo bajo"
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
    n_municipios = len(perfil)
    df = perfil.copy()
    df["equipos"] = 0

    if n_equipos_original >= n_municipios * 3:
        df["equipos"] = 1
        n_restantes = n_equipos_original - n_municipios
    else:
        n_restantes = n_equipos_original

    df["equipos"] += (
        (df["score"] / df["score"].sum() * n_restantes)
        .round().astype(int)
    )

    diferencia = n_equipos_original - df["equipos"].sum()
    if diferencia > 0:
        df.loc[df["score"].idxmax(), "equipos"] += diferencia
    elif diferencia < 0:
        df.loc[df["score"].idxmin(), "equipos"] += diferencia

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