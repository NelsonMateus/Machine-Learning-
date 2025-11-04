import streamlit as st
import xgboost as xgb
import pandas as pd

st.title("DETECTOR DE NOTAS FALSAS - NELSON")
st.write("Por Nelson - Cientista de Dados")

@st.cache_resource
def carregar_modelo():
    model = xgb.XGBClassifier()
    model.load_model('detector_notas_nelson.json')  # Streamlit encontra automaticamente na mesma pasta
    return model

model = carregar_modelo()

st.header("Meça a nota e digite:")
col1, col2 = st.columns(2)

with col1:
    length = st.number_input("Comprimento (LENGTH, mm)", value=214.8)
    left = st.number_input("Margem esquerda (LEFT, mm)", value=130.0)
    right = st.number_input("Margem direita (RIGHT, mm)", value=130.0)

with col2:
    bottom = st.number_input("Margem inferior (BOTTOM, mm)", value=9.0)
    top = st.number_input("Margem superior (TOP, mm)", value=10.0)

if st.button("VERIFICAR NOTA"):
    dados = [[length, left, right, bottom, top]]
    pred = model.predict(dados)[0]
    prob = model.predict_proba(dados)[0][1]
    
    st.subheader("RESULTADO:")
    if pred == 1:
        st.error(f"**FALSA!** (Probabilidade: {prob:.1%})")
        if bottom > 9.8:
            st.warning("ALERTA: Margem inferior muito alta!")
    else:
        st.success(f"**GENUÍNA!** (Probabilidade de falsa: {prob:.1%})")

st.markdown("---")
st.caption("Modelo treinado com XGBoost | Dados: 200 notas | Acurácia: 98.8%")
