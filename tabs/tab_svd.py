# tabs/tab_svd.py
import time
import numpy as np
import streamlit as st
import plotly.graph_objects as go

PFX = "svd"

def _draw_heatmap(Mk, k):
    fig = go.Figure(data=go.Heatmap(z=Mk, colorscale="Viridis", zsmooth=False))
    fig.update_layout(
        template="plotly_white",
        height=520,
        margin=dict(l=10, r=10, t=40, b=10),
        title=f"rank-{k} 근사"
    )
    return fig

def render():
    st.subheader("SVD: 랭크-k 근사로 복원 (한 번 재생)")

    # 1) 데이터 행렬 만들기
    n = st.slider("크기 n", 20, 60, 32, key=f"{PFX}:n")
    x = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(x, x)
    # 부드러운 패턴 예시
    M = np.sin(3 * X) * np.cos(4 * Y) + 0.3 * np.outer(np.sin(2 * x), np.ones_like(x))

    # 2) SVD
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    kmax = len(S)

    # 3) 미리보기 파라미터 + 애니메이션 제어
    c1, c2, c3 = st.columns(3)
    with c1:
        k_preview = st.slider("표시할 rank k", 1, kmax, min(10, kmax), 1, key=f"{PFX}:k")
    with c2:
        fps = st.slider("FPS", 2, 30, 12, key=f"{PFX}:fps")
    with c3:
        anim_until = st.slider("애니메이션 마지막 k", 1, kmax, kmax, 1, key=f"{PFX}:kend")

    # 4) 정적 미리보기
    holder = st.empty()
    Mk_prev = (U[:, :k_preview] * S[:k_preview]) @ Vt[:k_preview, :]
    holder.plotly_chart(_draw_heatmap(Mk_prev, k_preview), use_container_width=True)

    # 5) 버튼 1회 재생
    if st.button("🎬 rank-1 → rank-k 한 번 재생", key=f"{PFX}:once"):
        st.session_state[f"{PFX}:playing"] = True

    if st.session_state.get(f"{PFX}:playing", False):
        kend = int(anim_until)
        for k in range(1, kend + 1):
            Mk = (U[:, :k] * S[:k]) @ Vt[:k, :]
            holder.plotly_chart(_draw_heatmap(Mk, k), use_container_width=True)
            time.sleep(1.0 / max(1, fps))
        st.session_state[f"{PFX}:playing"] = False
