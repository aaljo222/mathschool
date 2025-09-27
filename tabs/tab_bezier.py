import numpy as np, streamlit as st, plotly.graph_objects as go
from utils.anim import playbar, step_loop

PFX = "bezier4"

def _bezier(P, t):
    # De Casteljau (cubic)
    A = (1-t)*P[0] + t*P[1]
    B = (1-t)*P[1] + t*P[2]
    C = (1-t)*P[2] + t*P[3]
    D = (1-t)*A + t*B
    E = (1-t)*B + t*C
    F = (1-t)*D + t*E
    return F, (A,B,C,D,E)

def render():
    st.subheader("Cubic Bézier (De Casteljau)")

    cols = st.columns(4)
    defaults = [(0.0,0.0),(0.3,0.8),(0.7,-0.6),(1.0,0.1)]
    P = []
    for i,c in enumerate(cols):
        with c:
            x = st.slider(f"P{i}x", -1.5, 1.5, float(defaults[i][0]), 0.01, key=f"{PFX}:x{i}")
            y = st.slider(f"P{i}y", -1.0, 1.0,  float(defaults[i][1]), 0.01, key=f"{PFX}:y{i}")
            P.append(np.array([x,y]))
    P = np.array(P)

    fps  = st.slider("FPS", 2, 30, 12, key=f"{PFX}:fps")
    secs = st.slider("길이(초)", 1, 12, 6, key=f"{PFX}:secs")
    steps = max(2, int(secs*fps))

    holder = st.empty()
    playing = playbar(PFX)

    def draw(t):
        F,(A,B,C,D,E) = _bezier(P, t)
        # 곡선 샘플
        T  = np.linspace(0,1,300)
        C2 = np.array([_bezier(P, tt)[0] for tt in T])

        fig = go.Figure()
        fig.add_scatter(x=P[:,0], y=P[:,1], mode="lines+markers", name="control")
        fig.add_scatter(x=C2[:,0], y=C2[:,1], mode="lines", name="curve", line=dict(width=3))
        fig.add_scatter(x=[F[0]], y=[F[1]], mode="markers", name="point", marker=dict(size=10))
        # 보조선
        fig.add_scatter(x=[A[0],B[0],C[0]], y=[A[1],B[1],C[1]], mode="lines", name="level1")
        fig.add_scatter(x=[D[0],E[0]], y=[D[1],E[1]], mode="lines", name="level2")
        fig.update_layout(template="plotly_white", height=520,
                          xaxis=dict(range=[-1.6,1.6], scaleanchor="y", scaleratio=1),
                          yaxis=dict(range=[-1.2,1.2]))
        holder.plotly_chart(fig, use_container_width=True)

    if playing:
        for k in step_loop(steps, fps=fps, key=PFX):
            draw(k/(steps-1))
    else:
        draw(0.35)
