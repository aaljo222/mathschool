# tabs/tab_conic.py
import numpy as np
import streamlit as st
from utils.plot import contour_implicit

def render():
    st.subheader("포물선 / 쌍곡선 시각화 (Implicit Contour)")

    shape = st.radio("곡선 선택", ["포물선 (Parabola)", "쌍곡선 (Hyperbola)"], horizontal=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        h = st.slider("중심 h", -5.0, 5.0, 0.0, 0.1)
        k = st.slider("중심 k", -5.0, 5.0, 0.0, 0.1)
    with col2:
        a = st.slider("a", 0.5, 5.0, 2.0, 0.1)
        b = st.slider("b (쌍곡선에서 사용)", 0.5, 5.0, 1.5, 0.1)
    with col3:
        x_min, x_max = st.slider("x-범위", -10, 10, (-6, 6))
        y_min, y_max = st.slider("y-범위", -10, 10, (-6, 6))

    if shape.startswith("포물선"):
        F = lambda X, Y: Y - (a*(X - h)**2 + k)
        fig = contour_implicit(F, (x_min, x_max), (y_min, y_max), title="포물선")
    else:
        F = lambda X, Y: ((X - h)**2)/(a**2) - ((Y - k)**2)/(b**2) - 1
        fig = contour_implicit(F, (x_min, x_max), (y_min, y_max), title="쌍곡선")

    st.plotly_chart(fig, use_container_width=True)
