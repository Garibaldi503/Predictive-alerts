import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

st.set_page_config(page_title="06 — Predictive Overstock & Stockout Alerts", layout="wide")

# --------------------------
# Demo Data Generator
# --------------------------
@st.cache_data
def load_demo_data():
    np.random.seed(42)
    dates = pd.date_range(end=datetime.today(), periods=60)
    skus = [f"SKU-{i}" for i in range(1, 6)]
    data = []
    for sku in skus:
        demand = np.random.poisson(lam=20, size=len(dates))
        on_hand = np.random.randint(50, 200)
        for d, q in zip(dates, demand):
            data.append([sku, d, q, on_hand, 50, 7])  # product_id, date, quantity, on_hand, unit_cost, lead_time_days
    return pd.DataFrame(data, columns=["product_id", "date", "quantity", "on_hand", "unit_cost", "lead_time_days"])

# --------------------------
# Load Data Section
# --------------------------
st.title("Predictive Overstock & Stockout Alerts")
st.markdown("Upload your CSV or use demo data to get started.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file:
    df_raw = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully.")
else:
    df_raw = load_demo_data()
    st.info("Using built-in demo data.")

# --------------------------
# Field Mapping
# --------------------------
st.subheader("Field Mapping")
columns = df_raw.columns.tolist()

field_map = {}
required_fields = {
    "product_id": "Product/SKU ID",
    "date": "Date",
    "quantity": "Quantity Sold",
    "on_hand": "On Hand Stock",
}
optional_fields = {
    "unit_cost": "Unit Cost",
    "lead_time_days": "Lead Time (Days)"
}

for key, label in {**required_fields, **optional_fields}.items():
    field_map[key] = st.selectbox(f"{label}:", options=["-- None --"] + columns, index=(columns.index(key)+1) if key in columns else 0)

# --------------------------
# Data Cleaning
# --------------------------
df = pd.DataFrame()
for key in required_fields.keys():
    if field_map[key] == "-- None --":
        st.error(f"Please map the field for: {required_fields[key]}")
        st.stop()
    else:
        df[key] = df_raw[field_map[key]]

for key in optional_fields.keys():
    if field_map[key] != "-- None --":
        df[key] = df_raw[field_map[key]]

# Ensure correct formats
df["date"] = pd.to_datetime(df["date"])
df.sort_values(["product_id", "date"], inplace=True)

st.success("Field mapping complete and data prepared.")
st.dataframe(df.head())

# --------------------------
# Simple Forecast Logic (Mean Demand)
# --------------------------
st.subheader("Forecast & Alerts")
lead_time = st.number_input("Default Lead Time (Days):", min_value=1, max_value=60, value=7)
service_level = st.slider("Service Level (%):", 50, 99, 95)

alerts = []
for sku, grp in df.groupby("product_id"):
    avg_demand = grp["quantity"].mean()
    reorder_point = avg_demand * lead_time
    safety_stock = avg_demand * (service_level/100)
    current_stock = grp["on_hand"].iloc[-1]

    if current_stock < reorder_point:
        status = "Stockout Risk"
    elif current_stock > reorder_point + safety_stock:
        status = "Overstock"
    else:
        status = "OK"

    alerts.append([sku, current_stock, reorder_point, safety_stock, status])

alerts_df = pd.DataFrame(alerts, columns=["SKU", "Current Stock", "Reorder Point", "Safety Stock", "Status"])
st.dataframe(alerts_df)

# --------------------------
# Download Alerts CSV
# --------------------------
csv = alerts_df.to_csv(index=False).encode('utf-8')
st.download_button("Download Alerts CSV", csv, "predictive_alerts.csv", "text/csv")

