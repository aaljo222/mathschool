# tabs/tab_lissajous.py
import numpy as np, streamlit as st, plotly.graph_objects as go
from utils.anim import playbar, step_loop

def render():
    st.subheader("리사주 곡선 (두 주기의 비와 위상차)")
    a = st.slider("fₓ (정수)", 1, 10, 3)
    b = st.slider("fᵧ (정수)", 1, 10, 2)
    phi = np.deg2rad(st.slider("위상차 (deg)", 0, 180, 90))
    secs = st.slider("길이(초)", 1, 8, 4)
    fps  = st.slider("FPS", 5, 30, 20)
    playing = playbar("lissajous")

    ph = st.empty()
    T = 2*np.pi
    def draw(tmax):
        t = np.linspace(0, tmax, 1000)
        x = np.sin(a*t)
        y = np.sin(b*t + phi)
        fig = go.Figure(go.Scatter(x=x, y=y, mode="lines"))
        fig.update_layout(template="plotly_white", height=520,
                          xaxis=dict(range=[-1.2,1.2]), yaxis=dict(range=[-1.2,1.2],
                          scaleanchor="x", scaleratio=1))
        ph.plotly_chart(fig, use_container_width=True)

    if playing:
        for k in step_loop(secs*fps, fps=fps, key="lissajous"):
            draw((k+1)/fps*2*np.pi)
    else:
        draw(2*np.pi)
