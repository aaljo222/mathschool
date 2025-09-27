# tabs/tab_bezier.py
import numpy as np, streamlit as st, plotly.graph_objects as go
from utils.anim import playbar, step_loop

def _interp(P, t):  # 한 단계 선형보간
    return (1-t)*P[:-1] + t*P[1:]

def render():
    st.subheader("Bézier 곡선 (de Casteljau 알고리즘)")
    deg = st.radio("차수", [2,3], horizontal=True)
    pts = []
    cols = st.columns(deg+1)
    defaults = [(0,0), (0.4,0.8), (0.8,-0.2), (1.1,0.6)]
    for i in range(deg+1):
        with cols[i]:
            x = st.number_input(f"P{i}x", value=float(defaults[i][0]))
            y = st.number_input(f"P{i}y", value=float(defaults[i][1]))
            pts.append([x,y])
    P0 = np.array(pts, float)
    playing = playbar("bezier"); ph = st.empty()

    def draw(t):
        fig = go.Figure()
        # control polygon
        fig.add_trace(go.Scatter(x=P0[:,0], y=P0[:,1], mode="lines+markers", name="control"))
        # de Casteljau ladder
        P = P0.copy()
        while len(P)>1:
            P = _interp(P, t)
            fig.add_trace(go.Scatter(x=P[:,0], y=P[:,1], mode="lines+markers", showlegend=False))
        fig.update_layout(template="plotly_white", height=500,
                          xaxis=dict(range=[-0.2,1.4]), yaxis=dict(range=[-1,1], scaleanchor="x", scaleratio=1))
        ph.plotly_chart(fig, use_container_width=True)

    if playing:
        for i in step_loop(60, fps=24, key="bezier"):
            draw(i/59)
    else:
        draw(0.6)
