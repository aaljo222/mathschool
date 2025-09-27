import numpy as np, streamlit as st, plotly.graph_objects as go
from utils.anim import playbar, step_loop

PFX = "svd_show"

def render():
    st.subheader("SVD 저랭크 근사 (열/행 모드)")

    n = st.slider("행렬 크기 n (n×n)", 20, 60, 32, key=f"{PFX}:n")
    fps  = st.slider("FPS", 2, 30, 10, key=f"{PFX}:fps")
    secs = st.slider("길이(초)", 1, 12, 6, key=f"{PFX}:secs")

    # 예제 행렬 생성 (부드러운 패턴)
    i = np.linspace(0, 2*np.pi, n)
    j = np.linspace(0, 2*np.pi, n)
    X,Y = np.meshgrid(i, j, indexing="ij")
    A = (np.sin(2*X)+0.6*np.cos(3*Y)+0.4*np.sin(X+Y))

    U,S,Vt = np.linalg.svd(A, full_matrices=False)
    rank = len(S)

    playing = playbar(PFX)
    holder = st.empty()
    steps  = max(2, int(secs*fps))

    def draw(t):
        k = max(1, int(1 + t*(rank-1)))
        Ak = (U[:,:k] * S[:k]) @ Vt[:k,:]
        fig = go.Figure(data=go.Heatmap(z=Ak, colorscale="RdBu"))
        fig.update_layout(template="plotly_white", height=520,
                          title=f"rank-{k} 근사  (전체 rank={rank})")
        holder.plotly_chart(fig, use_container_width=True)

    if playing:
        for m in step_loop(steps, fps=fps, key=PFX):
            draw(m/(steps-1))
    else:
        draw(0.2)
