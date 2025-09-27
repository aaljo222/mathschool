# tabs/tab_wave_interference.py
import numpy as np, streamlit as st, plotly.graph_objects as go
from utils.anim import playbar, step_loop

def render():
    st.subheader("두 사인파의 간섭 (비트)")
    f1 = st.slider("f₁ (Hz)", 1.0, 10.0, 5.0, 0.1)
    f2 = st.slider("f₂ (Hz)", 1.0, 10.0, 6.0, 0.1)
    tmax = st.slider("표시 구간 (초)", 0.5, 5.0, 2.0, 0.1)
    fps = st.slider("FPS", 5, 30, 20)
    playing = playbar("beats")
    ph = st.empty()

    def draw(t0):
        t = np.linspace(t0, t0+tmax, 1200)
        y1 = np.sin(2*np.pi*f1*t)
        y2 = np.sin(2*np.pi*f2*t)
        y  = y1 + y2
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=y1, name="y1", opacity=0.5))
        fig.add_trace(go.Scatter(x=t, y=y2, name="y2", opacity=0.5))
        fig.add_trace(go.Scatter(x=t, y=y,  name="합성", line=dict(width=3)))
        fig.update_layout(template="plotly_white", height=480,
                          xaxis_title="t", yaxis_title="amplitude")
        ph.plotly_chart(fig, use_container_width=True)

    if playing:
        for k in step_loop(120, fps=fps, key="beats"):
            draw(k/fps*0.3)
    else:
        draw(0.0)
