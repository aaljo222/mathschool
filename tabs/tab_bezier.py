# tabs/tab_bezier.py
import numpy as np, streamlit as st, plotly.graph_objects as go
from utils.anim import next_frame_index

PFX = "bezier"

def render():
    st.subheader("3차 베지에 · De Casteljau")

    def pt(lbl, x, y):
        c = st.columns(2)
        return np.array([c[0].slider(f"{lbl}x",-2.0,2.0,x,0.1,key=f"{PFX}:{lbl}x"),
                         c[1].slider(f"{lbl}y",-2.0,2.0,y,0.1,key=f"{PFX}:{lbl}y")])

    P0 = pt("P0", -1.5, -1.0)
    P1 = pt("P1", -0.2,  1.2)
    P2 = pt("P2",  0.8, -0.8)
    P3 = pt("P3",  1.6,  0.9)

    fps   = st.slider("FPS", 2, 30, 15, key=f"{PFX}:fps")
    steps = st.slider("프레임 수", 20, 200, 120, key=f"{PFX}:steps")
    autorun = st.checkbox("자동 재생", True, key=f"{PFX}:auto")

    k = next_frame_index(PFX, steps, fps, autorun)
    t = k / max(1, steps-1)

    # 곡선 샘플
    T = np.linspace(0,1,300)[:,None]
    B = (1-T)**3*P0 + 3*(1-T)**2*T*P1 + 3*(1-T)*T**2*P2 + T**3*P3

    # De Casteljau
    A = (1-t)*P0 + t*P1
    Bm= (1-t)*P1 + t*P2
    C = (1-t)*P2 + t*P3
    D = (1-t)*A  + t*Bm
    E = (1-t)*Bm + t*C
    S = (1-t)*D  + t*E  # curve point

    fig = go.Figure()
    # control polygon
    fig.add_scatter(x=[P0[0],P1[0],P2[0],P3[0]], y=[P0[1],P1[1],P2[1],P3[1]],
                    mode="lines+markers", name="control", line=dict(dash="dash"))
    # curve
    fig.add_scatter(x=B[:,0], y=B[:,1], mode="lines", name="Bezier", line=dict(width=3))
    # intermediates
    for (X,Y,nm) in [([A[0],Bm[0],C[0]],[A[1],Bm[1],C[1]],"level1"),
                     ([D[0],E[0]],[D[1],E[1]],"level2")]:
        fig.add_scatter(x=X, y=Y, mode="lines+markers", name=nm, showlegend=False)
    fig.add_scatter(x=[S[0]], y=[S[1]], mode="markers", name="point", marker=dict(size=10))
    fig.update_layout(template="plotly_white", height=520,
                      xaxis=dict(range=[-2.2,2.2], zeroline=True),
                      yaxis=dict(range=[-2.0,2.0], zeroline=True, scaleanchor="x", scaleratio=1),
                      title=f"t={t:.2f}")
    st.plotly_chart(fig, use_container_width=True)
