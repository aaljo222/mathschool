# tabs/tab_taylor.py
import math
import numpy as np
import streamlit as st
from utils.plot import line_fig

def render():
    st.subheader("테일러 시리즈 근사")
    funcs_T = {"sin x": (lambda x: np.sin(x), "sin"),
               "cos x": (lambda x: np.cos(x), "cos"),
               "e^x":   (lambda x: np.exp(x), "exp")}
    fname = st.selectbox("함수 선택", list(funcs_T.keys()))
    f_true, kind = funcs_T[fname]
    c = st.slider("전개 중심 c", -2.0, 2.0, 0.0, 0.1)
    order = st.slider("차수 N", 0, 20, 6, 1)
    x_min, x_max = st.slider("x-범위", -10.0, 10.0, (-6.0, 6.0), 0.5)
    X = np.linspace(x_min, x_max, 1200)

    def taylor_series(kind, X, c, N):
        Ts = np.zeros_like(X)
        if kind == "exp":
            for n in range(N+1): Ts += (math.exp(c)/math.factorial(n)) * (X - c)**n
        elif kind == "sin":
            derivs = [np.sin, np.cos, lambda z:-np.sin(z), lambda z:-np.cos(z)]
            for n in range(N+1): Ts += (derivs[n%4](c) / math.factorial(n)) * (X - c)**n
        elif kind == "cos":
            derivs = [np.cos, lambda z:-np.sin(z), lambda z:-np.cos(z), np.sin]
            for n in range(N+1): Ts += (derivs[n%4](c) / math.factorial(n)) * (X - c)**n
        return Ts

    T = taylor_series(kind, X, c, order)
    fig_T = line_fig(X, [f_true(X), T], ["원함수", f"테일러 근사 (N={order})"], f"{fname} Taylor around c={c}")
    st.plotly_chart(fig_T, use_container_width=True)
