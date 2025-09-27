# tabs/tab_eigen2d.py
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from utils.anim import playbar, step_loop

PFX = "eigen2d"  # 이 탭만의 접두사

def render():
    st.subheader("2×2 선형변환 : 그리드 변형과 고유벡터")

    # --- 컨트롤 (모두 고유 key 부여) ---
    fps  = st.slider("FPS", 2, 20, 8, key=f"{PFX}_fps")
    secs = st.slider("길이(초)", 1, 8, 3, key=f"{PFX}_secs")
    a, b = st.slider("1행 [a, b]", -2.0, 2.0, (1.2, 0.4), 0.1, key=f"{PFX}_row1")
    c, d = st.slider("2행 [c, d]", -2.0, 2.0, (0.0, 0.9), 0.1, key=f"{PFX}_row2")
    playing = playbar(PFX)  # 재생/일시정지 UI (중복 버튼 만들지 않음)

    A = np.array([[a, b], [c, d]])
    placeholder = st.empty()

    def draw(t: float):
        # I → A 보간
        M = (1 - t) * np.eye(2) + t * A

        grid = np.linspace(-1, 1, 13)
        fig = go.Figure()

        # 원래/변형 그리드
        for g in grid:
            P1 = np.column_stack([grid, np.full_like(grid, g)])
            P2 = np.column_stack([np.full_like(grid, g), grid])
            Q1 = P1 @ M.T
            Q2 = P2 @ M.T

            fig.add_trace(go.Scatter(x=P1[:, 0], y=P1[:, 1], mode="lines",
                                     line=dict(color="#dddddd"), showlegend=False))
            fig.add_trace(go.Scatter(x=P2[:, 0], y=P2[:, 1], mode="lines",
                                     line=dict(color="#dddddd"), showlegend=False))
            fig.add_trace(go.Scatter(x=Q1[:, 0], y=Q1[:, 1], mode="lines",
                                     name="변형 그리드", showlegend=False))
            fig.add_trace(go.Scatter(x=Q2[:, 0], y=Q2[:, 1], mode="lines",
                                     name="변형 그리드", showlegend=False))

        # 고유벡터(실수부만 표시)
        try:
            vals, vecs = np.linalg.eig(A)
            for i in range(2):
                v = np.real(vecs[:, i])
                v = v / (np.linalg.norm(v) + 1e-12)  # 방향만
                fig.add_trace(go.Scatter(x=[0, v[0]], y=[0, v[1]],
                                         mode="lines+markers",
                                         name=f"eigvec {i+1}",
                                         line=dict(width=3)))
        except Exception:
            pass

        fig.update_layout(
            template="plotly_white",
            height=520,
            xaxis=dict(range=[-2.5, 2.5], zeroline=True),
            yaxis=dict(range=[-2.5, 2.5], zeroline=True,
                       scaleanchor="x", scaleratio=1),
            margin=dict(l=10, r=10, t=10, b=10),
        )
        placeholder.plotly_chart(fig, use_container_width=True)

    # --- 재생 루프 ---
    n_frames = max(2, int(secs * fps))
    if playing:
        for k in step_loop(n_frames, fps=fps, key=PFX):
            t = k / (n_frames - 1)
            draw(t)
    else:
        draw(1.0)
