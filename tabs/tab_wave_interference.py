# tabs/tab_wave_interference.py
import numpy as np, streamlit as st, plotly.graph_objects as go
from utils.anim import next_frame_index

PFX = "wave2"

def render():
    st.subheader("두 사인파 간섭")

    col = st.columns(4)
    with col[0]: f1 = st.slider("f₁ (Hz)", 0.5, 6.0, 2.0, 0.1, key=f"{PFX}:f1")
    with col[1]: f2 = st.slider("f₂ (Hz)", 0.5, 6.0, 2.2, 0.1, key=f"{PFX}:f2")
    with col[2]: A  = st.slider("진폭", 0.2, 2.0, 1.0, 0.1, key=f"{PFX}:A")
    with col[3]: autorun = st.checkbox("자동 재생", True, key=f"{PFX}:auto")

    fps = st.slider("FPS", 2, 30, 15, key=f"{PFX}:fps")
    steps = st.slider("프레임 수", 20, 200, 120, key=f"{PFX}:steps")

    k = next_frame_index(PFX, steps, fps, autorun)
    t0 = k / max(1, steps-1) * 2*np.pi

    x = np.linspace(0, 2*np.pi, 800)
    y1 = A*np.sin(f1*x - t0)
    y2 = A*np.sin(f2*x - t0)
    y  = y1 + y2

    fig = go.Figure()
    fig.add_scatter(x=x, y=y1, mode="lines", name="y₁")
    fig.add_scatter(x=x, y=y2, mode="lines", name="y₂")
    fig.add_scatter(x=x, y=y,  mode="lines", name="합성", line=dict(width=3))
    fig.update_layout(template="plotly_white", height=480, title=f"위상 t₀={t0:.2f} rad")
    st.plotly_chart(fig, use_container_width=True)
