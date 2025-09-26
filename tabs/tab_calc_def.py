# tabs/tab_calc_def.py
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from utils.plot import line_fig

def render():
    st.subheader("미분·적분 (정의)")
    funcs = {
        "x^2": (lambda x: x**2, lambda x: 2*x, lambda a,b: (b**3 - a**3)/3),
        "3x^3 + 6x": (lambda x: 3*x**3 + 6*x, lambda x: 9*x**2 + 6, lambda a,b: (3/4)*(b**4-a**4)+3*(b**2-a**2)),
        "sin x": (lambda x: np.sin(x), lambda x: np.cos(x), lambda a,b: -np.cos(b)+np.cos(a)),
        "e^x": (lambda x: np.exp(x), lambda x: np.exp(x), lambda a,b: np.exp(b)-np.exp(a)),
    }
    f_name = st.selectbox("함수 선택", list(funcs.keys()), index=1)
    f, fprime, Fint = funcs[f_name]

    st.markdown("### 미분 (극한 정의)  $\\lim_{h\\to 0}\\frac{f(a+h)-f(a)}{h}$")
    c1, c2 = st.columns(2)
    with c1:
        a0 = st.slider("미분 지점 a", -3.0, 3.0, 1.0, 0.1)
        h = st.slider("증분 h", 1e-5, 0.5, 0.1, 0.0001, format="%.5f")
        deriv_est = (f(a0+h) - f(a0)) / h
        deriv_true = fprime(a0)
        st.write(f"수치 미분 ≈ **{deriv_est:.6f}**,  해석 미분 = **{deriv_true:.6f}**")
    with c2:
        xs = np.linspace(-3, 3, 800)
        ys = f(xs)
        tangent = f(a0) + deriv_true*(xs-a0)
        fig_d = line_fig(xs, [ys, tangent], [f_name, "접선(해석 기울기)"], "함수와 접선")
        fig_d.add_trace(go.Scatter(x=[a0], y=[f(a0)], mode="markers", name="a"))
        st.plotly_chart(fig_d, use_container_width=True)

    st.markdown("### 정적분 (리만 합)  $\\int_a^b f(x)\\,dx$")
    c3, c4 = st.columns(2)
    with c3:
        A_int, B_int = st.slider("구간 [a, b]", -3.0, 3.0, (-1.0, 2.0), 0.1)
        N = st.slider("직사각형 개수 N", 2, 400, 30, 2)
        xs = np.linspace(A_int, B_int, N+1)
        mids = (xs[:-1] + xs[1:]) / 2
        dx = (B_int - A_int)/N
        approx = np.sum(f(mids) * dx)
        exact = Fint(A_int, B_int)
        st.write(f"리만합 ≈ **{approx:.6f}**,  해석적 값 = **{exact:.6f}**,  오차 = **{approx-exact:.6e}**")
    with c4:
        X = np.linspace(A_int, B_int, 1000); Y = f(X)
        fig_i = go.Figure()
        fig_i.add_trace(go.Scatter(x=X, y=Y, mode="lines", name=f_name))
        for i in range(N):
            x0, x1 = xs[i], xs[i+1]; xm = (x0+x1)/2; y = f(xm)
            fig_i.add_shape(type="rect", x0=x0, x1=x1, y0=0, y1=y,
                            line=dict(width=1), fillcolor="LightSkyBlue", opacity=0.2)
        fig_i.update_layout(title="리만 합(중점 규칙)", template="plotly_white")
        st.plotly_chart(fig_i, use_container_width=True)
