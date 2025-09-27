# tabs/tab_eigen2d.py
import numpy as np, streamlit as st, plotly.graph_objects as go
from utils.anim import playbar, step_loop

PFX = "eigen2d"  # 이 탭 전용 접두사(모든 key에 붙여 중복 방지)

def render():
    st.subheader("2×2 선형변환 : 그리드 변형과 고유벡터")

    # ── 컨트롤(모두 고유 key 사용) ───────────────────────────
    fps  = st.slider("FPS", 2, 30, 12, key=f"{PFX}:fps")
    secs = st.slider("길이(초)", 1, 10, 4,   key=f"{PFX}:secs")
    a,b  = st.slider("A의 1행 [a b]", -2.0, 2.0, (1.2, 0.4), 0.1, key=f"{PFX}:row1")
    c,d  = st.slider("A의 2행 [c d]", -2.0, 2.0, (0.0, 0.9), 0.1, key=f"{PFX}:row2")
    A = np.array([[a,b],[c,d]])

    playing = playbar(PFX)           # 재생/일시정지 토글(상태 기억)
    holder  = st.empty()
    steps   = max(2, int(secs*fps))  # 총 프레임 수

    def draw(t: float):
        # I → A 선형 보간
        M = (1 - t) * np.eye(2) + t * A
        grid = np.linspace(-1, 1, 13)
        fig = go.Figure()

        # 원래 격자(회색) + 변환된 격자(진한 선)
        for g in grid:
            P1 = np.c_[grid, np.full_like(grid, g)]
            P2 = np.c_[np.full_like(grid, g), grid]
            Q1, Q2 = P1 @ M.T, P2 @ M.T

            fig.add_trace(go.Scatter(x=P1[:,0], y=P1[:,1], mode="lines",
                                     line=dict(color="#e5e5e5"), showlegend=False))
            fig.add_trace(go.Scatter(x=P2[:,0], y=P2[:,1], mode="lines",
                                     line=dict(color="#e5e5e5"), showlegend=False))
            fig.add_trace(go.Scatter(x=Q1[:,0], y=Q1[:,1], mode="lines",
                                     line=dict(width=2), showlegend=False))
            fig.add_trace(go.Scatter(x=Q2[:,0], y=Q2[:,1], mode="lines",
                                     line=dict(width=2), showlegend=False))

        # 고유벡터(실수일 때만)
        try:
            vals, vecs = np.linalg.eig(A)
            for i in range(2):
                v = vecs[:, i].real
                fig.add_trace(go.Scatter(x=[0, v[0]], y=[0, v[1]],
                                         mode="lines+markers",
                                         name=f"eigvec {i+1} (λ={vals[i]:.2g})",
                                         line=dict(width=3)))
        except Exception:
            pass

        fig.update_layout(
            template="plotly_white", height=520,
            xaxis=dict(range=[-2.5, 2.5], zeroline=True),
            yaxis=dict(range=[-2.5, 2.5], zeroline=True, scaleanchor="x", scaleratio=1),
        )
        holder.plotly_chart(fig, use_container_width=True)

    if playing:
        for k in step_loop(steps, fps=fps, key=PFX):
            draw(k / (steps - 1))
    else:
        draw(1.0)
