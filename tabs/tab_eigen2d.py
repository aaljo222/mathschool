# tabs/tab_eigen2d.py
import numpy as np, streamlit as st, plotly.graph_objects as go
from utils.anim import next_frame_index

PFX = "eigen2d"

def render():
    st.subheader("2×2 선형변환 : 그리드 변형과 고유벡터")

    a,b = st.slider("1행 [a b]", -2.0, 2.0, (1.2, 0.4), 0.1, key=f"{PFX}:r1")
    c,d = st.slider("2행 [c d]", -2.0, 2.0, (0.0, 0.9), 0.1, key=f"{PFX}:r2")
    A = np.array([[a,b],[c,d]])

    col1, col2, col3 = st.columns([1,1,1])
    with col1: steps = st.slider("프레임 수", 20, 120, 60, key=f"{PFX}:steps")
    with col2: fps   = st.slider("FPS", 2, 30, 12, key=f"{PFX}:fps")
    with col3: autorun = st.checkbox("자동 재생", True, key=f"{PFX}:auto")

    k = next_frame_index(PFX, steps, fps, autorun)
    t = k / max(1, steps-1)  # 0→1 보간
    M = (1-t)*np.eye(2) + t*A

    grid = np.linspace(-1.8, 1.8, 11)
    fig = go.Figure()
    for g in grid:
        P = np.vstack([np.c_[grid, np.full_like(grid,g)],
                       np.c_[np.full_like(grid,g), grid]])
        Q = P @ M.T
        fig.add_scatter(x=P[:len(grid),0], y=P[:len(grid),1], mode="lines", line=dict(color="#dddddd"), showlegend=False)
        fig.add_scatter(x=P[len(grid):,0], y=P[len(grid):,1], mode="lines", line=dict(color="#dddddd"), showlegend=False)
        fig.add_scatter(x=Q[:len(grid),0], y=Q[:len(grid),1], mode="lines", name="변환격자", line=dict(width=2))
        fig.add_scatter(x=Q[len(grid):,0], y=Q[len(grid):,1], mode="lines", line=dict(width=2), showlegend=False)

    # eigenvectors of A (정지표시)
    try:
        vals, vecs = np.linalg.eig(A)
        for i in range(2):
            v = vecs[:, i].real
            v = v / (np.linalg.norm(v) + 1e-12)
            fig.add_scatter(x=[0, v[0]], y=[0, v[1]], mode="lines+markers",
                            name=f"eigvec {i+1} (λ={vals[i].real:.2f})", line=dict(width=4))
    except Exception:
        pass

    lim = 2.5
    fig.update_layout(template="plotly_white", height=540,
                      xaxis=dict(range=[-lim, lim], zeroline=True),
                      yaxis=dict(range=[-lim, lim], zeroline=True, scaleanchor="x", scaleratio=1),
                      title=f"보간 t={t:.2f}  (I → A)")
    st.plotly_chart(fig, use_container_width=True)
