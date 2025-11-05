import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Econometría Financiera Básica", layout="wide")
st.title("📊 Aplicación de Econometría Financiera Básica")
st.markdown("Evalúa tus conocimientos con datos financieros simulados o con tus propios datos.")

# Sidebar
st.sidebar.header("Opciones")
opcion_datos = st.sidebar.radio("¿Qué datos deseas usar?", ("Generar datos aleatorios", "Subir mis propios datos"))

# Función para generar datos aleatorios
def generar_datos_financieros(n=100):
    np.random.seed()
    rendimiento_activo = np.random.normal(0.001, 0.02, n)
    tasa_interes = np.random.normal(0.02, 0.005, n) + np.random.normal(0, 0.002, n)
    precio_activo = 100 * np.exp(np.cumsum(rendimiento_activo))
    return pd.DataFrame({
        "Rendimiento": rendimiento_activo,
        "Tasa_Interes": tasa_interes,
        "Precio_Activo": precio_activo
    })

# Cargar datos iniciales
if opcion_datos == "Generar datos aleatorios":
    n_muestra = st.sidebar.slider("Tamaño de la muestra", 30, 1000, 100)
    df = generar_datos_financieros(n_muestra)
else:
    uploaded_file = st.sidebar.file_uploader("Sube tu archivo CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        st.warning("Por favor sube un archivo CSV para continuar.")
        st.stop()

# Inicializar estado
if "df" not in st.session_state:
    st.session_state.df = df

# Botón para regenerar datos
if opcion_datos == "Generar datos aleatorios":
    if st.sidebar.button("🎲 Generar nuevos datos aleatorios"):
        st.session_state.df = generar_datos_financieros(n_muestra)
        st.rerun()

# Mostrar datos
st.subheader("📋 Vista previa de los datos")
st.dataframe(st.session_state.df.head())

# 1. Análisis descriptivo
st.header("1. Análisis Descriptivo")
desc = st.session_state.df.describe().T
desc['mediana'] = st.session_state.df.median()
st.dataframe(desc[['mean', 'mediana', 'std', 'min', 'max', '25%', '50%', '75%']])

# Diagrama de caja
st.subheader("Diagrama de Caja")
columna = st.selectbox("Selecciona una variable para el diagrama de caja", st.session_state.df.columns)
fig, ax = plt.subplots()
sns.boxplot(y=st.session_state.df[columna], ax=ax)
st.pyplot(fig)

# 2. Análisis de correlaciones
st.header("2. Análisis de Correlaciones")
numeric_cols = st.session_state.df.select_dtypes(include=np.number).columns
corr = st.session_state.df[numeric_cols].corr()

fig, ax = plt.subplots()
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# Gráfico de dispersión
st.subheader("Gráfico de Dispersión con Línea de Tendencia")
x_col = st.selectbox("Variable X", numeric_cols)
y_col = st.selectbox("Variable Y", numeric_cols, index=1)

fig, ax = plt.subplots()
sns.regplot(x=st.session_state.df[x_col], y=st.session_state.df[y_col], ax=ax, line_kws={"color": "red"})
ax.set_title(f"Relación entre {x_col} y {y_col}")
st.pyplot(fig)

# 3. Regresión simple
st.header("3. Regresión Lineal Simple")
X = st.session_state.df[[x_col]].values
y = st.session_state.df[y_col].values

model = LinearRegression()
model.fit(X, y)
beta = model.coef_[0]
intercepto = model.intercept_
r2 = model.score(X, y)

st.write(f"**Intercepto (α):** {intercepto:.4f}")
st.write(f"**Coeficiente (β):** {beta:.4f}")
st.write(f"**R²:** {r2:.4f}")
