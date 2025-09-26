# tabs/tab_trig.py
import numpy as np
import streamlit as st
from utils.plot import line_fig

def render():
    st.subheader("삼각함수: sin, cos (주파수/위상 조절)")
    col1, col2, col3 = st.columns(3)
    with col1: f = st.slider("주파수 f (Hz)", 0.1, 5.0, 1.0, 0.1)
    with col2: A = st.slider("진폭 A", 0.5, 3.0, 1.0, 0.1)
    with col3: phi = st.slider("위상 φ (라디안)", -np.pi, np.pi, 0.0, 0.1)

    t = np.linspace(0, 2, 1000)
    y_sin = A*np.sin(2*np.pi*f*t + phi)
    y_cos = A*np.cos(2*np.pi*f*t + phi)
    fig = line_fig(t, [y_sin, y_cos], ["A·sin(2πft+φ)", "A·cos(2πft+φ)"], "삼각함수")
    st.plotly_chart(fig, use_container_width=True)
