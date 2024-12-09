import streamlit as st
import joblib
import numpy as np

model = joblib.load('ML_kategorikendaraan.pkl')
encoder = joblib.load('encoder.pkl')

st.title("jenis-jenis kendaraan")
_id = st.number_input("ID: ", min_value=0, step=1, value=25)
Uraian = st.selectbox("Uraian:", options=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17], format_func=lambda x: ["Sedan", "Jeep", "Station Wagon", "Mini Cab", "Ambulance", "Truk/Kontainer", "Double Cabin", "Barang", "Trailer", "Truk Tangki", "Pemadam Api", "Traktor", "Pick Up", "Lain-Lain", "Bus Biasa", "Mini Bus", "Scooter", "SPM 50 CC keatas"][x])
Satuan = st.selectbox("Satuan : ", options=[0], format_func=lambda x: ["unit"][x]) 
Negara = st.number_input("Negara :", min_value=0, step=1, value=1044)
Swasta = st.selectbox("Swasta :", [0, 76, 1691, 440, 248, 803, 2, 10, 4, 17129, 176, 15])
Jenis_Kendaraan = st.selectbox("Jenis Kendaraan: ", options=[0, 1, 2, 3], format_func=lambda x: ["Mobil Penumpang", "Bis", "Sepeda Motor", "Mobil Barang"][x])
Tahun = st.selectbox("Tahun : ", [2023])

if st.button("jenis"):
    # Create a 2D array with shape (1, n_features)
    data = np.array([_id, Uraian, Satuan, Negara, Swasta, Jenis_Kendaraan, Tahun]).reshape(1, -1)
    try:
        pred_label = model.predict(data)[0]
        pred_jenis_kendaraan = encoder.inverse_transform([pred_label])[0]
        st.success(f"jenis kendaraan ini adalah: {pred_jenis_kendaraan}")
    except Exception as e:
        st.error(f"Error {e}")