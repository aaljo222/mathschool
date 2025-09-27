# tabs/tab_eigen2d.py
import numpy as np, streamlit as st, plotly.graph_objects as go
from utils.anim import playbar, step_loop

def render():
    st.subheader("2×2 선형변환 : 그리드 변형과 고유벡터")
    a,b = st.slider("1행", -2.0, 2.0, (1.2, 0.4), 0.1)
    c,d = st.slider("2행", -2.0, 2.0, (0.0, 0.9), 0.1)
    A = np.array([[a,b],[c,d]])
    fps = st.slider("FPS", 2, 20, 8)
    playing = playbar("eig2d")
    ph = st.empty()

    def draw(t):
        M = (1-t)*np.eye(2) + t*A   # I → A 보간
        grid = np.linspace(-1,1,13)
        fig = go.Figure()
        for g in grid:
            P = np.vstack([np.array([grid, np.full_like(grid,g)]).T,
                           np.array([np.full_like(grid,g), grid]).T])
            Q = P @ M.T
            fig.add_trace(go.Scatter(x=P[:len(grid),0], y=P[:len(grid),1], mode="lines",
                                     line=dict(color="#ddd")))
            fig.add_trace(go.Scatter(x=P[len(grid):,0], y=P[len(grid):,1], mode="lines",
                                     line=dict(color="#ddd")))
            fig.add_trace(go.Scatter(x=Q[:len(grid),0], y=Q[:len(grid),1], mode="lines"))
            fig.add_trace(go.Scatter(x=Q[len(grid):,0], y=Q[len(grid):,1], mode="lines"))

        # eigenvectors
        try:
            vals, vecs = np.linalg.eig(A)
            for i in range(2):
                v = vecs[:,i].real
                fig.add_trace(go.Scatter(x=[0, v[0]], y=[0, v[1]], mode="lines+markers",
                                         name=f"eigvec {i+1}", line=dict(width=3)))
        except Exception:
            pass

        fig.update_layout(template="plotly_white", height=520,
                          xaxis=dict(range=[-2.5,2.5], zeroline=True),
                          yaxis=dict(range=[-2.5,2.5], zeroline=True, scaleanchor="x", scaleratio=1))
        ph.plotly_chart(fig, use_container_width=True)

    if playing:
        for i in step_loop(40, fps=fps, key="eig2d"):
            draw(i/39)
    else:
        draw(1.0)
