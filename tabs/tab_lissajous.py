import numpy as np, streamlit as st, plotly.graph_objects as go
from utils.anim import playbar, step_loop

PFX = "lissajous"

def render():
    st.subheader("리사주 곡선")

    c1, c2, c3 = st.columns(3)
    with c1:
        a = st.slider("a", 1, 10, 3, key=f"{PFX}:a")
    with c2:
        b = st.slider("b", 1, 10, 2, key=f"{PFX}:b")
    with c3:
        delta = st.slider("δ (rad)", 0.0, 2*np.pi, np.pi/2, 0.01, key=f"{PFX}:d")
    fps = st.slider("FPS", 2, 30, 12, key=f"{PFX}:fps")
    secs = st.slider("길이(초)", 1, 12, 6, key=f"{PFX}:secs")

    holder = st.empty()
    steps = max(2, int(secs*fps))
    playing = playbar(PFX)

    T = np.linspace(0, 2*np.pi, 1200)
    X = np.sin(a*T + delta); Y = np.sin(b*T)

    def draw(t):
        n = max(10, int(t*len(T)))
        fig = go.Figure()
        fig.add_scatter(x=X[:n], y=Y[:n], mode="lines", line=dict(width=3), name="curve")
        fig.add_scatter(x=[X[n-1]], y=[Y[n-1]], mode="markers", name="tip")
        fig.update_layout(template="plotly_white", height=520,
                          xaxis=dict(scaleanchor="y", scaleratio=1, range=[-1.1,1.1]),
                          yaxis=dict(range=[-1.1,1.1]))
        holder.plotly_chart(fig, use_container_width=True)

    if playing:
        for k in step_loop(steps, fps=fps, key=PFX):
            draw(k/(steps-1))
    else:
        draw(0.25)
