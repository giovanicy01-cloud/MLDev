import streamlit as st
import pandas as pd
import pickle
import numpy as np
import joblib
import plotly.graph_objects as go

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Breast Cancer Pred",
    page_icon="🎗️",
    layout="wide"
)

# --- LOAD MODEL & SCALER ---
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('breast_cancer_model.pkl')
        scaler = joblib.load('scaler_kanker.pkl')
        return model, scaler
    except FileNotFoundError:
        st.error("File .pkl tidak ditemukan. Pastikan 'breast_cancer_model.pkl' dan 'scaler.pkl' ada di folder yang sama.")
        return None, None

model, scaler = load_assets()

# --- UI SIDEBAR (INPUT PARAMETER) ---
st.sidebar.header("⚙️ Input Parameter Sel")
st.sidebar.write("Geser slider untuk memasukkan nilai fitur sel dari hasil lab.")

def user_input_features():
    input_dict = {}
    
    # Agar tampilan rapi, kita bagi input menjadi 3 grup (Mean, SE, Worst)
    # Ini sesuai dengan struktur dataset aslinya (30 fitur)
    
    with st.sidebar.expander("1. Fitur Rata-rata (Mean)", expanded=True):
        input_dict['radius_mean'] = st.slider('Radius (Mean)', 6.0, 30.0, 14.0)
        input_dict['texture_mean'] = st.slider('Texture (Mean)', 9.0, 40.0, 19.0)
        input_dict['perimeter_mean'] = st.slider('Perimeter (Mean)', 43.0, 190.0, 90.0)
        input_dict['area_mean'] = st.slider('Area (Mean)', 143.0, 2500.0, 650.0)
        input_dict['smoothness_mean'] = st.slider('Smoothness (Mean)', 0.05, 0.17, 0.1)
        input_dict['compactness_mean'] = st.slider('Compactness (Mean)', 0.01, 0.35, 0.1)
        input_dict['concavity_mean'] = st.slider('Concavity (Mean)', 0.0, 0.43, 0.08)
        input_dict['concave points_mean'] = st.slider('Concave Points (Mean)', 0.0, 0.2, 0.05)
        input_dict['symmetry_mean'] = st.slider('Symmetry (Mean)', 0.1, 0.31, 0.18)
        input_dict['fractal_dimension_mean'] = st.slider('Fractal Dim (Mean)', 0.04, 0.1, 0.06)

    # Untuk fitur SE dan Worst, demi simplifikasi demo kita set default rata-rata 
    # Namun kita buat slider tersembunyi jika user ingin mengubahnya (Advanced Mode)
    with st.sidebar.expander("2. Fitur Error & Worst (Advanced)"):
        # Standard Error
        input_dict['radius_se'] = st.slider('Radius SE', 0.1, 3.0, 0.4)
        input_dict['texture_se'] = st.slider('Texture SE', 0.3, 5.0, 1.2)
        input_dict['perimeter_se'] = st.slider('Perimeter SE', 0.7, 22.0, 2.8)
        input_dict['area_se'] = st.slider('Area SE', 6.0, 550.0, 40.0)
        input_dict['smoothness_se'] = st.slider('Smoothness SE', 0.0, 0.03, 0.007)
        input_dict['compactness_se'] = st.slider('Compactness SE', 0.0, 0.14, 0.025)
        input_dict['concavity_se'] = st.slider('Concavity SE', 0.0, 0.4, 0.03)
        input_dict['concave points_se'] = st.slider('Concave Points SE', 0.0, 0.06, 0.01)
        input_dict['symmetry_se'] = st.slider('Symmetry SE', 0.0, 0.08, 0.02)
        input_dict['fractal_dimension_se'] = st.slider('Fractal Dim SE', 0.0, 0.03, 0.003)
        
        # Worst
        input_dict['radius_worst'] = st.slider('Radius Worst', 7.0, 36.0, 16.0)
        input_dict['texture_worst'] = st.slider('Texture Worst', 12.0, 50.0, 25.0)
        input_dict['perimeter_worst'] = st.slider('Perimeter Worst', 50.0, 252.0, 107.0)
        input_dict['area_worst'] = st.slider('Area Worst', 185.0, 4255.0, 880.0)
        input_dict['smoothness_worst'] = st.slider('Smoothness Worst', 0.07, 0.23, 0.13)
        input_dict['compactness_worst'] = st.slider('Compactness Worst', 0.02, 1.06, 0.25)
        input_dict['concavity_worst'] = st.slider('Concavity Worst', 0.0, 1.26, 0.27)
        input_dict['concave points_worst'] = st.slider('Concave Points Worst', 0.0, 0.3, 0.11)
        input_dict['symmetry_worst'] = st.slider('Symmetry Worst', 0.15, 0.67, 0.29)
        input_dict['fractal_dimension_worst'] = st.slider('Fractal Dim Worst', 0.05, 0.21, 0.08)

    features = pd.DataFrame(input_dict, index=[0])
    return features

# --- HALAMAN UTAMA ---
st.title("🎗️ Breast Cancer Diagnostic Dashboard")
st.markdown("""
Aplikasi ini menggunakan **Machine Learning (Random Forest)** untuk memprediksi apakah sel kanker payudara bersifat **Jinak (Benign)** atau **Ganas (Malignant)** berdasarkan karakteristik sel.
""")

st.divider()

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📋 Input Data")
    st.info("Silakan masukkan parameter hasil biopsi pada panel sebelah kiri.")
    input_df = user_input_features()
    
    # Tampilkan ringkasan input
    st.write("Preview Input (Mean Features):")
    st.dataframe(input_df.iloc[:, :5].T, height=200)

if model is not None and scaler is not None:
    # --- PREDIKSI ---
    # 1. Scale input data
    input_scaled = scaler.transform(input_df)
    
    # 2. Predict
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)
    
    diagnosis = "Malignant (Ganas)" if prediction[0] == 1 else "Benign (Jinak)"
    probability_malignant = prediction_proba[0][1]
    probability_benign = prediction_proba[0][0]

    # --- HASIL PREDIKSI (KANAN) ---
    with col2:
        st.subheader("🔍 Hasil Diagnosis AI")
        
        # Card untuk hasil
        if prediction[0] == 1:
            st.error(f"### Prediksi: {diagnosis}")
            color = "red"
        else:
            st.success(f"### Prediksi: {diagnosis}")
            color = "green"
            
        st.write(f"Model memiliki keyakinan sebesar **{max(probability_malignant, probability_benign)*100:.2f}%** terhadap keputusan ini.")

        # --- VISUALISASI GAUGE CHART ---
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = probability_malignant * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Probabilitas Keganasan (%)"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkred" if prediction[0]==1 else "lightgrey"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgreen"},
                    {'range': [50, 100], 'color': "salmon"}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50}}))
        
        st.plotly_chart(fig, use_container_width=True)

        # --- INTERPRETASI ---
        st.markdown("---")
        st.subheader("ℹ️ Interpretasi Cepat")
        if prediction[0] == 1:
            st.markdown("""
            ⚠️ **Perhatian:** Model mendeteksi pola yang mirip dengan sel kanker ganas.
            * Nilai **Area**, **Perimeter**, atau **Concavity** mungkin lebih tinggi dari batas normal.
            * Disarankan untuk melakukan konsultasi medis lebih lanjut.
            """)
        else:
            st.markdown("""
            ✅ **Aman:** Model mendeteksi pola sel yang cenderung jinak.
            * Struktur sel terlihat teratur dan ukurannya dalam batas wajar.
            * Tetap lakukan pemeriksaan rutin.
            """)

else:
    st.warning("Menunggu file model diupload...")