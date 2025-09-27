# tabs/tab_svd.py
import numpy as np, streamlit as st, plotly.graph_objects as go
from utils.anim import next_frame_index

PFX = "svd"

def render():
    st.subheader("SVD: 랭크-k 근사로 복원")

    n = st.slider("크기 n", 20, 60, 32, key=f"{PFX}:n")
    # 예제 행렬 생성(부드러운 패턴)
    x = np.linspace(-1,1,n)
    X,Y = np.meshgrid(x,x)
    M = np.sin(3*X) * np.cos(4*Y) + 0.3*np.outer(np.sin(2*x), np.ones_like(x))

    U,S,Vt = np.linalg.svd(M, full_matrices=False)
    kmax = len(S)

    col1, col2, col3 = st.columns(3)
    with col1: steps = st.slider("프레임 수", 5, kmax, min(20,kmax), key=f"{PFX}:steps")
    with col2: fps   = st.slider("FPS", 2, 30, 12, key=f"{PFX}:fps")
    with col3: autorun = st.checkbox("자동 재생", True, key=f"{PFX}:auto")

    k = 1 + next_frame_index(PFX, steps, fps, autorun) * (kmax-1) // max(1, steps-1)

    Mk = (U[:, :k] * S[:k]) @ Vt[:k, :]
    fig = go.Figure(data=go.Heatmap(z=Mk, colorscale="Viridis"))
    fig.update_layout(template="plotly_white", height=520, title=f"rank-{k} 근사")
    st.plotly_chart(fig, use_container_width=True)
