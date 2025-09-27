import numpy as np, streamlit as st, plotly.graph_objects as go
from utils.anim import playbar, step_loop

PFX = "waveint"

def render():
    st.subheader("두 파동의 간섭")

    c1, c2, c3 = st.columns(3)
    with c1:
        f1 = st.slider("f₁(Hz)", 0.1, 10.0, 2.0, 0.1, key=f"{PFX}:f1")
        A1 = st.slider("A₁", 0.0, 2.0, 1.0, 0.1, key=f"{PFX}:A1")
    with c2:
        f2 = st.slider("f₂(Hz)", 0.1, 10.0, 2.5, 0.1, key=f"{PFX}:f2")
        A2 = st.slider("A₂", 0.0, 2.0, 1.0, 0.1, key=f"{PFX}:A2")
    with c3:
        phase = st.slider("위상차 φ (rad)", 0.0, 2*np.pi, np.pi/3, 0.01, key=f"{PFX}:phi")
        fps   = st.slider("FPS", 2, 30, 12, key=f"{PFX}:fps")
    secs = st.slider("길이(초)", 1, 10, 5, key=f"{PFX}:secs")

    x = np.linspace(0, 2*np.pi, 600)
    holder = st.empty()
    playing = playbar(PFX)
    steps = max(2, int(secs*fps))

    def draw(t):
        # 진행파: 시간에 따라 위상 이동
        y1 = A1*np.sin(f1*x - 2*np.pi*t)
        y2 = A2*np.sin(f2*x - 2*np.pi*t + phase)
        ys = y1 + y2
        fig = go.Figure()
        fig.add_scatter(x=x, y=y1, mode="lines", name="y₁")
        fig.add_scatter(x=x, y=y2, mode="lines", name="y₂")
        fig.add_scatter(x=x, y=ys, mode="lines", name="y₁+y₂", line=dict(width=3))
        fig.update_layout(template="plotly_white", height=450,
                          xaxis_title="x", yaxis_title="amplitude")
        holder.plotly_chart(fig, use_container_width=True)

    if playing:
        for k in step_loop(steps, fps=fps, key=PFX):
            draw(k/(steps-1))
    else:
        draw(0.0)
