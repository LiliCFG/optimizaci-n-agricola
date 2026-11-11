import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import statsmodels.formula.api as smf
import streamlit as st

# Configuración inicial del dashboard
st.set_page_config(
    page_title="Dashboard Agrotecnológico", page_icon="🌱", layout="wide"
)

st.title(
    "🌱 Optimización Inferencial del Rendimiento Agrícola y Eficiencia de Riego"
)
st.caption(
    "Proyecto Integrador - Materia: Análisis y Exploración de Datos (INC-2505)"
)


# ==========================================
# HITO 1: INGESTA DE DATOS CON DUCKDB
# ==========================================
@st.cache_data
def cargar_datos():
  con = duckdb.connect(database=":memory:")
  con.execute("SELECT setseed(0.42);")
  con.execute("""
        CREATE TABLE telemetria AS
        WITH base AS (
            SELECT 
                range AS parcela_id,
                ROUND(100.0 + random() * 200.0, 2) AS volumen_riego,
                ROUND(50.0 + random() * 100.0, 2) AS fertilizante_kg,
                CASE WHEN random() < 0.50 THEN 'Tratamiento_A' ELSE 'Tratamiento_B' END AS tipo_tratamiento,
                (-0.8 + random() * 1.6) AS ruido
            FROM range(1, 101)
        )
        SELECT 
            parcela_id, volumen_riego, fertilizante_kg, tipo_tratamiento,
            ROUND(5.0 + (0.035 * volumen_riego) + (0.020 * fertilizante_kg) + 
            (CASE WHEN tipo_tratamiento = 'Tratamiento_B' THEN 1.8 ELSE 0.0 END) + ruido, 2) AS rendimiento_ton
        FROM base;
    """)
  return con.execute("SELECT * FROM telemetria").df()


df = cargar_datos()

# ==========================================
# HITO 2 Y 3: MODELO OLS Y PRUEBA T-STUDENT
# ==========================================
# Regresión Lineal OLS
modelo_ols = smf.ols(
    "rendimiento_ton ~ volumen_riego + fertilizante_kg", data=df
).fit()

# Prueba t-Student para evaluar tratamientos A vs B
grupo_a = df[df["tipo_tratamiento"] == "Tratamiento_A"]["rendimiento_ton"]
grupo_b = df[df["tipo_tratamiento"] == "Tratamiento_B"]["rendimiento_ton"]
t_stat, p_val = stats.ttest_ind(grupo_b, grupo_a, alternative="greater")

# ==========================================
# CONTROLES Y FILTROS LATERALES
# ==========================================
st.sidebar.header("⚙️ Filtros del Invernadero")
tratamiento_sel = st.sidebar.multiselect(
    "Seleccionar Tratamiento:",
    options=df["tipo_tratamiento"].unique(),
    default=df["tipo_tratamiento"].unique(),
)

df_filtrado = df[df["tipo_tratamiento"].isin(tratamiento_sel)]

# ==========================================
# HITOS 4 Y 5: TARJETAS KPI Y VISUALIZACIÓN
# ==========================================
col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
col_kpi1.metric(
    "Rendimiento Promedio Global",
    f"{df_filtrado['rendimiento_ton'].mean():.2f} t/ha",
)
col_kpi2.metric("R² Ajustado del Modelo", f"{modelo_ols.rsquared_adj:.2%}")
col_kpi3.metric("P-Valor Prueba t (B vs A)", f"{p_val:.4e}")

st.markdown("---")

# Cuadrícula 2x2 para las 4 gráficas requeridas
row1_col1, row1_col2 = st.columns(2)
row2_col1, row2_col2 = st.columns(2)

# Gráfica 1: Heatmap
with row1_col1:
  st.subheader("1. Mapa de Calor de Correlación")
  fig, ax = plt.subplots(figsize=(5, 3.5))
  corr = df_filtrado[
      ["volumen_riego", "fertilizante_kg", "rendimiento_ton"]
  ].corr()
  sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
  st.pyplot(fig)

# Gráfica 2: Regresión OLS
with row1_col2:
  st.subheader("2. Rendimiento vs. Volumen de Riego")
  fig, ax = plt.subplots(figsize=(5, 3.5))
  sns.regplot(
      data=df_filtrado,
      x="volumen_riego",
      y="rendimiento_ton",
      ax=ax,
      color="green",
      line_kws={"color": "red"},
  )
  st.pyplot(fig)

# Gráfica 3: Boxplot Comparativo
with row2_col1:
  st.subheader("3. Comparativa de Tratamientos (Boxplot)")
  fig, ax = plt.subplots(figsize=(5, 3.5))
  sns.boxplot(
      data=df_filtrado,
      x="tipo_tratamiento",
      y="rendimiento_ton",
      palette="Set2",
      ax=ax,
  )
  st.pyplot(fig)

# Gráfica 4: Histograma de Residuos
with row2_col2:
  st.subheader("4. Diagnóstico de Residuos (Errores)")
  residuos = modelo_ols.resid
  fig, ax = plt.subplots(figsize=(5, 3.5))
  sns.histplot(residuos, kde=True, color="purple", ax=ax)
  ax.set_title("Distribución Gaussiana de Errores")
  st.pyplot(fig)