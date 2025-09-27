# tabs/tab_lissajous.py
import numpy as np, streamlit as st, plotly.graph_objects as go
from utils.anim import next_frame_index

PFX = "liss"

def render():
    st.subheader("리사주 곡선")

    col = st.columns(4)
    with col[0]: a = st.slider("a", 1, 9, 3, key=f"{PFX}:a")
    with col[1]: b = st.slider("b", 1, 9, 2, key=f"{PFX}:b")
    with col[2]: d = st.slider("δ (rad)", 0.0, np.pi, np.pi/2, 0.01, key=f"{PFX}:d")
    with col[3]: autorun = st.checkbox("자동 재생", True, key=f"{PFX}:auto")

    fps   = st.slider("FPS", 2, 30, 12, key=f"{PFX}:fps")
    steps = st.slider("프레임 수", 20, 200, 120, key=f"{PFX}:steps")

    k = next_frame_index(PFX, steps, fps, autorun)
    tt = np.linspace(0, 2*np.pi, 1200)
    x, y = np.sin(a*tt + d), np.sin(b*tt)
    t = k / max(1, steps-1)
    px, py = np.sin(a*(t*2*np.pi) + d), np.sin(b*(t*2*np.pi))

    fig = go.Figure()
    fig.add_scatter(x=x, y=y, mode="lines", name="곡선", line=dict(width=2))
    fig.add_scatter(x=[px], y=[py], mode="markers", name="점", marker=dict(size=10))
    fig.update_layout(template="plotly_white", height=520,
                      xaxis=dict(range=[-1.1,1.1], zeroline=True),
                      yaxis=dict(range=[-1.1,1.1], zeroline=True, scaleanchor="x", scaleratio=1),
                      title=f"t={t:.2f}")
    st.plotly_chart(fig, use_container_width=True)
