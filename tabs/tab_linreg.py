# tabs/tab_linreg.py
import numpy as np
import streamlit as st
import plotly.graph_objects as go

def render():
    st.subheader("선형회귀: y = ax + b (경사하강)")
    c1, c2, c3 = st.columns(3)
    with c1:
        true_a = st.slider("진짜 기울기 (데이터 생성)", -5.0, 5.0, 2.0, 0.1)
        true_b = st.slider("진짜 절편 (데이터 생성)", -10.0, 10.0, -1.0, 0.1)
    with c2:
        noise = st.slider("노이즈 표준편차", 0.0, 5.0, 1.0, 0.1)
        npts = st.slider("데이터 개수", 10, 300, 80, 10)
    with c3:
        lr = st.slider("학습률(learning rate)", 1e-5, 0.5, 0.05, 0.0005, format="%.5f")
        steps = st.slider("경사하강 스텝 수", 1, 2000, 200, 10)

    rng = np.random.default_rng(0)
    x = np.linspace(-5, 5, npts)
    y = true_a*x + true_b + rng.normal(0, noise, size=npts)

    def mse(a, b): return np.mean((y - (a*x + b))**2)
    def grad(a, b):
        n = len(x); e = (a*x + b) - y
        return (1.0/n)*np.sum(e*x), (1.0/n)*np.sum(e)

    a_hat, b_hat = 0.0, 0.0
    history = []
    for _ in range(steps):
        da, db = grad(a_hat, b_hat)
        a_hat -= lr*da; b_hat -= lr*db
        history.append(mse(a_hat, b_hat))

    fig_lr = go.Figure()
    fig_lr.add_trace(go.Scatter(x=x, y=y, mode="markers", name="data"))
    fig_lr.add_trace(go.Scatter(x=x, y=true_a*x+true_b, mode="lines", name="true line"))
    fig_lr.add_trace(go.Scatter(x=x, y=a_hat*x+b_hat, mode="lines", name="learned line"))
    fig_lr.update_layout(title=f"학습 결과: a≈{a_hat:.3f}, b≈{b_hat:.3f}", template="plotly_white")

    fig_mse = go.Figure()
    fig_mse.add_trace(go.Scatter(x=np.arange(1, steps+1), y=history, mode="lines", name="MSE"))
    fig_mse.update_layout(title="MSE 감소", xaxis_title="step", yaxis_title="MSE", template="plotly_white")

    p1, p2 = st.columns(2)
    with p1: st.plotly_chart(fig_lr, use_container_width=True)
    with p2: st.plotly_chart(fig_mse, use_container_width=True)
