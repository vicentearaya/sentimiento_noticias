import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from transformers import pipeline

st.set_page_config(page_title="Análisis de Sentimiento Electoral", page_icon="🗳️", layout="wide")

st.title("🗳️ Análisis de Sentimiento Electoral")
st.caption("Análisis de sentimiento de comentarios sobre candidatos presidenciales usando BERT multilingüe.")

@st.cache_resource(show_spinner=False)
def load_model():
    # Modelo multilingüe de 1-5 estrellas; lo convertimos a Pos/Neu/Neg
    clf = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
    return clf

def map_label_to_polarity(label: str):
    # Mapea '1 star'..'5 stars' a Negativo / Neutro / Positivo + score normalizado.
    stars = int(label.split()[0])
    if stars <= 2:
        return "Negativo", (stars - 1) / 1.0  # 0..1 dentro de negativo
    elif stars == 3:
        return "Neutro", 0.5
    else:
        return "Positivo", (stars - 3) / 2.0  # 0..1 dentro de positivo

st.sidebar.header("Entrada de datos")
mode = st.sidebar.radio("Seleccione modo de entrada", ["Pegar texto", "Cargar CSV", "Demo"], index=2)

if mode == "Pegar texto":
    st.subheader("Pegar comentarios (uno por línea)")
    text = st.text_area("Comentarios", height=200, placeholder="Escribe o pega comentarios aquí, uno por línea...")
    candidate_name = st.text_input("Candidato asociado", placeholder="Ej: Evelyn Matthei")
    
    if st.button("Analizar"):
        comments = [t.strip() for t in text.splitlines() if t.strip()]
        if comments and candidate_name:
            clf = load_model()
            preds = clf(comments)
            rows = []
            for c, p in zip(comments, preds):
                pol, strength = map_label_to_polarity(p["label"])
                rows.append({
                    "comment": c,
                    "candidate": candidate_name,
                    "label_raw": p["label"],
                    "score_raw": p["score"],
                    "polarity": pol,
                    "strength": strength
                })
            df = pd.DataFrame(rows)
            st.session_state["df"] = df
        else:
            st.warning("Ingrese al menos un comentario y el nombre del candidato.")

elif mode == "Cargar CSV":
    st.subheader("Cargar CSV con columnas 'comment' y 'candidate' (en ese orden)")
    st.info("📋 Formato requerido: comment,candidate")
    file = st.file_uploader("Seleccionar CSV", type=["csv"])
    
    if file is not None:
        data = pd.read_csv(file)
        if "comment" not in data.columns or "candidate" not in data.columns:
            st.error("El CSV debe contener las columnas 'comment' y 'candidate'.")
        else:
            st.success(f"✅ Archivo cargado: {len(data)} comentarios encontrados")
            st.dataframe(data.head(), use_container_width=True)
            
            if st.button("Analizar CSV"):
                clf = load_model()
                with st.spinner("Analizando sentimientos..."):
                    preds = clf(data["comment"].astype(str).tolist())
                    rows = []
                    for idx, (c, cand, p) in enumerate(zip(data["comment"].astype(str).tolist(), 
                                                           data["candidate"].astype(str).tolist(), 
                                                           preds)):
                        pol, strength = map_label_to_polarity(p["label"])
                        rows.append({
                            "comment": c,
                            "candidate": cand,
                            "label_raw": p["label"],
                            "score_raw": p["score"],
                            "polarity": pol,
                            "strength": strength
                        })
                    df = pd.DataFrame(rows)
                    st.session_state["df"] = df

else:
    st.subheader("Demo con datos de ejemplo")
    st.info("⚠️ Asegúrate de tener el archivo 'data/comentarios.csv' con columnas: comment,candidate")
    
    if st.button("Analizar demo"):
        try:
            demo_df = pd.read_csv("data/comentarios.csv")
            if "comment" not in demo_df.columns or "candidate" not in demo_df.columns:
                st.error("El CSV demo debe tener columnas 'comment' y 'candidate'")
            else:
                clf = load_model()
                preds = clf(demo_df["comment"].tolist())
                rows = []
                for c, cand, p in zip(demo_df["comment"].tolist(), 
                                     demo_df["candidate"].tolist(), 
                                     preds):
                    pol, strength = map_label_to_polarity(p["label"])
                    rows.append({
                        "comment": c,
                        "candidate": cand,
                        "label_raw": p["label"],
                        "score_raw": p["score"],
                        "polarity": pol,
                        "strength": strength
                    })
                st.session_state["df"] = pd.DataFrame(rows)
        except FileNotFoundError:
            st.error("No se encontró el archivo 'data/comentarios.csv'")

# ============================================
# VISUALIZACIÓN DE RESULTADOS
# ============================================
df = st.session_state.get("df")
if df is not None and not df.empty:
    st.success(f"✅ Se analizaron {len(df)} comentarios.")
    
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Positivos", int((df["polarity"]=="Positivo").sum()))
    col2.metric("Neutros", int((df["polarity"]=="Neutro").sum()))
    col3.metric("Negativos", int((df["polarity"]=="Negativo").sum()))
    col4.metric("Promedio score raw", round(float(df["score_raw"].mean()), 3))

    with st.expander("Ver tabla de resultados"):
        st.dataframe(df, use_container_width=True)

    # ============================================
    # GRÁFICO 1: Distribución de sentimiento total
    # ============================================
    st.header("📊 Gráfico 1: Distribución de Sentimiento Total")
    
    counts = df["polarity"].value_counts().reset_index()
    counts.columns = ["polarity", "count"]
    
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        fig_bar = px.bar(
            counts, 
            x="polarity", 
            y="count", 
            title="Distribución de Polaridad (Barras)", 
            text="count",
            color="polarity",
            color_discrete_map={"Positivo": "#2ecc71", "Neutro": "#f39c12", "Negativo": "#e74c3c"}
        )
        fig_bar.update_traces(textposition='outside')
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col_g2:
        fig_pie = px.pie(
            counts, 
            names="polarity", 
            values="count", 
            title="Proporción por Polaridad (Pastel)", 
            hole=0.4,
            color="polarity",
            color_discrete_map={"Positivo": "#2ecc71", "Neutro": "#f39c12", "Negativo": "#e74c3c"}
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # ============================================
# GRÁFICO 2: Sentimiento por candidato
# ============================================
if df is not None and "candidate" in df.columns:
    st.header("📊 Gráfico 2: Sentimiento por Candidato")
    
    # Calcular porcentaje de positivos por candidato
    candidate_sentiment = df.groupby(['candidate', 'polarity']).size().unstack(fill_value=0)
    total_per_candidate = candidate_sentiment.sum(axis=1)
    
    # Asegurarse de que existe la columna Positivo
    if 'Positivo' not in candidate_sentiment.columns:
        candidate_sentiment['Positivo'] = 0
    
    candidate_sentiment['%_Positivo'] = (candidate_sentiment['Positivo'] / total_per_candidate * 100)
    candidate_sentiment = candidate_sentiment.reset_index()
    candidate_sentiment = candidate_sentiment.sort_values('%_Positivo', ascending=False)
    
    fig_candidates = px.bar(
        candidate_sentiment,
        x='candidate',
        y='%_Positivo',
        title='Porcentaje de Comentarios Positivos por Candidato',
        labels={'candidate': 'Candidato', '%_Positivo': '% Positivos'},
        text='%_Positivo',
        color='%_Positivo',
        color_continuous_scale='RdYlGn'
    )
    fig_candidates.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig_candidates.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_candidates, use_container_width=True)
    
    # Resumen por candidato
    st.subheader("📋 Resumen por Candidato")
    summary = df.groupby('candidate')['polarity'].value_counts().unstack(fill_value=0)
    summary['Total'] = summary.sum(axis=1)
    if 'Positivo' in summary.columns:
        summary['% Positivo'] = (summary['Positivo'] / summary['Total'] * 100).round(2)
    st.dataframe(summary, use_container_width=True)


st.markdown("---")
st.caption("📌 Nota: El modelo produce etiquetas de 1 a 5 estrellas. Se mapean a Negativo (1-2), Neutro (3), Positivo (4-5).")
st.caption("⚠️ Este análisis NO corresponde a una predicción electoral; solo refleja el sentimiento presente en los comentarios recolectados.")