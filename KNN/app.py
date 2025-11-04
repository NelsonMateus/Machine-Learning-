import streamlit as st
import pandas as pd
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Análise de Clusters", layout="wide")

# ============================
# 1. Carregar modelo e scaler
# ============================
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.title("💡 Análise de Clusters com K-Means")
st.write("Explore os grupos formados pelo modelo e visualize as métricas de agrupamento.")

# ============================
# 2. Upload de dados
# ============================
uploaded_file = st.file_uploader("📁 Envie um arquivo CSV com as mesmas variáveis usadas no treinamento", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Pré-visualização dos dados:")
    st.dataframe(df.head())

    # ============================
    # 3. Pré-processamento
    # ============================
    X_scaled = scaler.transform(df)

    # ============================
    # 4. Predição dos clusters
    # ============================
    df["cluster"] = model.predict(X_scaled)
    st.success("✅ Clusters atribuídos com sucesso!")
    st.write(df.head())

    # ============================
    # 5. Métrica Silhouette
    # ============================
    score = silhouette_score(X_scaled, df["cluster"])
    st.metric("Média do Silhouette", f"{score:.3f}")

    # ============================
    # 6. Heatmap das médias por cluster
    # ============================
    cluster_means = df.groupby("cluster").mean()
    cluster_means_scaled = (cluster_means - cluster_means.mean()) / cluster_means.std()

    st.subheader("🌡️ Heatmap das médias padronizadas por cluster")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(cluster_means_scaled, annot=True, cmap='coolwarm', center=0, ax=ax)
    st.pyplot(fig)

    # ============================
    # 7. Download com rótulos
    # ============================
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="💾 Baixar dados com clusters",
        data=csv,
        file_name='dados_clusterizados.csv',
        mime='text/csv'
    )
