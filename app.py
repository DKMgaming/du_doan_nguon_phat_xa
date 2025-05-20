import streamlit as st
import pandas as pd
import re
import math
import folium
from folium import Marker
from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error
import numpy as np
import joblib
from io import BytesIO

st.set_page_config(page_title="Dự đoán tọa độ nguồn phát xạ", layout="wide")
st.title("📡 Dự đoán tọa độ nguồn phát xạ từ dữ liệu trạm kiểm soát")

# Hàm xử lý tọa độ từ chuỗi dạng '10.421 N, 105.432 E'
def parse_coordinates(coord_str):
    try:
        match = re.match(r"([\d\.]+)\s*[\u00B0\sNn]?,?\s*([\d\.]+)\s*[\u00B0\sEe]?", coord_str)
        if match:
            lat, lon = float(match.group(1)), float(match.group(2))
            return lat, lon
        else:
            return None, None
    except:
        return None, None

# Hàm tính Azimuth giữa hai điểm địa lý
def calculate_azimuth(lat1, lon1, lat2, lon2):
    lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
    lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)
    d_lon = lon2_rad - lon1_rad
    x = math.sin(d_lon) * math.cos(lat2_rad)
    y = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(d_lon)
    azimuth_rad = math.atan2(x, y)
    azimuth_deg = (math.degrees(azimuth_rad) + 360) % 360
    return azimuth_deg

# Hàm tiền xử lý file Excel để tạo tập dữ liệu huấn luyện
def preprocess_xlsm(file):
    df = pd.read_excel(file)
    df = df.dropna(subset=["ControlDevicePosition1", "RadiationPosition1"])
    lat_rx, lon_rx, lat_tx, lon_tx = [], [], [], []

    for _, row in df.iterrows():
        lat1, lon1 = parse_coordinates(str(row["ControlDevicePosition1"]))
        lat2, lon2 = parse_coordinates(str(row["RadiationPosition1"]))
        if None not in (lat1, lon1, lat2, lon2):
            lat_rx.append(lat1)
            lon_rx.append(lon1)
            lat_tx.append(lat2)
            lon_tx.append(lon2)

    df_processed = pd.DataFrame({
        "Lat_RX": lat_rx,
        "Lon_RX": lon_rx,
        "Lat_TX": lat_tx,
        "Lon_TX": lon_tx
    })
    df_processed["Azimuth"] = df_processed.apply(
        lambda row: calculate_azimuth(row["Lat_RX"], row["Lon_RX"], row["Lat_TX"], row["Lon_TX"]), axis=1
    )
    return df_processed[["Lat_RX", "Lon_RX", "Azimuth", "Lat_TX", "Lon_TX"]]

uploaded_file = st.file_uploader("📤 Tải lên file dữ liệu Excel (.xlsm hoặc .xlsx)", type=["xlsm", "xlsx"])

if uploaded_file:
    with st.spinner("Đang xử lý dữ liệu..."):
        df = preprocess_xlsm(uploaded_file)
        st.success("✅ Dữ liệu đã được xử lý!")
        st.dataframe(df.head())

    if st.button("🚀 Huấn luyện mô hình"):
        X = df[["Lat_RX", "Lon_RX", "Azimuth"]]
        y_lat = df["Lat_TX"]
        y_lon = df["Lon_TX"]

        X_train, X_test, y_lat_train, y_lat_test, y_lon_train, y_lon_test = train_test_split(
            X, y_lat, y_lon, test_size=0.2, random_state=42
        )

        model_lat = RandomForestRegressor(n_estimators=100, random_state=42)
        model_lon = RandomForestRegressor(n_estimators=100, random_state=42)

        model_lat.fit(X_train, y_lat_train)
        model_lon.fit(X_train, y_lon_train)

        y_lat_pred = model_lat.predict(X_test)
        y_lon_pred = model_lon.predict(X_test)

        rmse_lat = np.sqrt(mean_squared_error(y_lat_test, y_lat_pred))
        rmse_lon = np.sqrt(mean_squared_error(y_lon_test, y_lon_pred))

        st.write(f"RMSE vĩ độ: {rmse_lat:.6f}")
        st.write(f"RMSE kinh độ: {rmse_lon:.6f}")


        joblib.dump(model_lat, "model_lat.joblib")
        joblib.dump(model_lon, "model_lon.joblib")
        st.success("✅ Đã huấn luyện và lưu mô hình thành công!")

    st.markdown("---")
    st.subheader("🔮 Dự đoán tọa độ nguồn phát xạ")
    col1, col2, col3 = st.columns(3)
    with col1:
        input_lat_rx = st.number_input("Vĩ độ trạm RX", format="%f")
    with col2:
        input_lon_rx = st.number_input("Kinh độ trạm RX", format="%f")
    with col3:
        input_azimuth = st.number_input("Góc phương vị (Azimuth)", min_value=0.0, max_value=360.0, format="%f")

    if st.button("📍 Dự đoán tọa độ TX"):
        try:
            model_lat = joblib.load("model_lat.joblib")
            model_lon = joblib.load("model_lon.joblib")
            input_df = pd.DataFrame([[input_lat_rx, input_lon_rx, input_azimuth]], columns=["Lat_RX", "Lon_RX", "Azimuth"])
            pred_lat = model_lat.predict(input_df)[0]
            pred_lon = model_lon.predict(input_df)[0]
            st.success(f"📡 Dự đoán tọa độ nguồn phát xạ: ({pred_lat:.6f}, {pred_lon:.6f})")

            m = folium.Map(location=[input_lat_rx, input_lon_rx], zoom_start=12)
            Marker([input_lat_rx, input_lon_rx], tooltip="Trạm RX", icon=folium.Icon(color='blue')).add_to(m)
            Marker([pred_lat, pred_lon], tooltip="Dự đoán TX", icon=folium.Icon(color='red')).add_to(m)
            st_folium(m, width=700, height=500)

            csv_data = pd.DataFrame({
                "Lat_RX": [input_lat_rx],
                "Lon_RX": [input_lon_rx],
                "Azimuth": [input_azimuth],
                "Lat_TX_DuDoan": [pred_lat],
                "Lon_TX_DuDoan": [pred_lon]
            })
            csv_buffer = BytesIO()
            csv_data.to_csv(csv_buffer, index=False)
            st.download_button("📥 Tải kết quả dự đoán", data=csv_buffer.getvalue(), file_name="du_doan_toa_do_TX.csv", mime="text/csv")

        except Exception as e:
            st.error(f"Lỗi khi tải mô hình hoặc dự đoán: {e}")
