# tabs/tab_fourier_series.py
import numpy as np, streamlit as st, plotly.graph_objects as go
from utils.anim import playbar, step_loop

def _target(kind, t):
    if kind == "사각파":
        return np.sign(np.sin(2*np.pi*t))
    elif kind == "톱니파":
        return 2*(t - np.floor(t+0.5))
    else:  # 삼각파
        return 2/np.pi*np.arcsin(np.sin(2*np.pi*t))

def _series(kind, t, N):
    y = np.zeros_like(t)
    if kind == "사각파":
        for k in range(1, 2*N, 2):
            y += (4/np.pi)*(1/k)*np.sin(2*np.pi*k*t)
    elif kind == "톱니파":
        for k in range(1, N+1):
            y += -(2/np.pi)*(1/k)*np.sin(2*np.pi*k*t)
    else:  # 삼각파
        for k in range(1, 2*N, 2):
            y += (8/np.pi**2)*(1/k**2)*np.sin(2*np.pi*k*t)*(-1)**((k-1)//2)
    return y

def render():
    st.subheader("푸리에 급수: 조화 성분을 차례로 추가")
    kind = st.selectbox("파형", ["사각파","삼각파","톱니파"])
    Nmax = st.slider("최대 고조파 수(N)", 1, 60, 20)
    t = np.linspace(0,1,1200,endpoint=False)
    y_true = _target(kind, t)

    playing = playbar("fourier")
    figph = st.empty()

    def draw(n):
        yN = _series(kind, t, max(1,n))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=y_true, name="목표", mode="lines"))
        fig.add_trace(go.Scatter(x=t, y=yN, name=f"부분합 N={n}", mode="lines"))
        fig.update_layout(template="plotly_white", height=460, xaxis_title="t", yaxis_title="y")
        figph.plotly_chart(fig, use_container_width=True)

    if playing:
        for i in step_loop(Nmax, fps=8, key="fourier"):
            draw(i+1)
    else:
        draw(Nmax)
