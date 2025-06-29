# streamlit_live.py
import streamlit as st
import pandas as pd
import numpy as np
import time

st.title("📈 Real-Time Equity Curve")
chart = st.line_chart()

equity = 10000
data = []

while True:
    equity += np.random.randn() * 10  # Simulate equity fluctuation
    data.append(equity)
    df = pd.DataFrame(data, columns=["Equity"])
    chart.line_chart(df)
    time.sleep(2)
