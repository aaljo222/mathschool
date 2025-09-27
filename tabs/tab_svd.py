# tabs/tab_svd.py
import numpy as np, streamlit as st, plotly.graph_objects as go
from utils.anim import playbar, step_loop

def _sample(n=120):
    # 간단한 합성 영상(원+사각형)
    X,Y = np.meshgrid(np.linspace(-1,1,n), np.linspace(-1,1,n))
    img = ( (X**2+Y**2 < 0.35**2).astype(float)*0.9 +
            ((abs(X)<0.6)&(abs(Y)<0.05)).astype(float)*0.6 )
    return img

def render():
    st.subheader("SVD로 이미지 랭크-k 근사")
    n = st.slider("해상도", 60, 180, 120, 20)
    A = _sample(n)
    U,S,Vt = np.linalg.svd(A, full_matrices=False)
    kmax = min(n, st.slider("최대 k", 1, min(n,60), 30))
    fps = st.slider("FPS", 2, 20, 8)
    playing = playbar("svd")
    ph = st.empty()

    def draw(k):
        Ak = (U[:,:k]*(S[:k])) @ Vt[:k,:]
        fig = go.Figure(data=go.Heatmap(z=Ak, colorscale="Gray"))
        fig.update_layout(template="plotly_white", height=520, xaxis_showticklabels=False, yaxis_autorange='reversed',
                          title=f"rank-{k} 근사 / 총 에너지 비율 ≈ {S[:k].sum()/S.sum():.3f}")
        ph.plotly_chart(fig, use_container_width=True)

    if playing:
        for i in step_loop(kmax, fps=fps, key="svd"):
            draw(i+1)
    else:
        draw(kmax)
