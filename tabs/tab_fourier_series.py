# tabs/tab_fourier_series.py
import numpy as np, streamlit as st, plotly.graph_objects as go
from utils.anim import next_frame_index

PFX = "fsq"

def render():
    st.subheader("푸리에 급수: 사각파 근사")

    Nmax = st.slider("최대 항 수", 1, 61, 31, step=2, key=f"{PFX}:Nmax")
    fps   = st.slider("FPS", 2, 30, 12, key=f"{PFX}:fps")
    steps = st.slider("프레임 수", 5, Nmax, min(21,Nmax), key=f"{PFX}:steps")
    autorun = st.checkbox("자동 재생", True, key=f"{PFX}:auto")

    k = 1 + 2*( next_frame_index(PFX, steps, fps, autorun) * ((Nmax-1)//2) // max(1, steps-1) )
    x = np.linspace(-np.pi, np.pi, 1200)
    y = np.zeros_like(x)
    for n in range(1, k+1, 2):  # 홀수항
        y += (4/np.pi) * (1/n) * np.sin(n*x)

    fig = go.Figure()
    fig.add_scatter(x=x, y=y, mode="lines", name=f"N={k}")
    fig.update_layout(template="plotly_white", height=480, title=f"사각파 근사 (홀수항 N={k})")
    st.plotly_chart(fig, use_container_width=True)
