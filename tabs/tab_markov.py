# tabs/tab_markov.py
import numpy as np, streamlit as st, plotly.graph_objects as go
from utils.anim import next_frame_index

PFX = "markov"

def render():
    st.subheader("마르코프 체인: 분포의 시간 전개")

    def _norm(r): s = sum(r) or 1.0; return [x/s for x in r]
    c = st.columns(3)
    r1 = [_norm([c[0].slider("p11",0.,1.,0.7,0.01,key=f"{PFX}:p11"),
                 c[1].slider("p12",0.,1.,0.2,0.01,key=f"{PFX}:p12"),
                 c[2].slider("p13",0.,1.,0.1,0.01,key=f"{PFX}:p13")])]
    c = st.columns(3)
    r2 = [_norm([c[0].slider("p21",0.,1.,0.1,0.01,key=f"{PFX}:p21"),
                 c[1].slider("p22",0.,1.,0.8,0.01,key=f"{PFX}:p22"),
                 c[2].slider("p23",0.,1.,0.1,0.01,key=f"{PFX}:p23")])]
    c = st.columns(3)
    r3 = [_norm([c[0].slider("p31",0.,1.,0.2,0.01,key=f"{PFX}:p31"),
                 c[1].slider("p32",0.,1.,0.2,0.01,key=f"{PFX}:p32"),
                 c[2].slider("p33",0.,1.,0.6,0.01,key=f"{PFX}:p33")])]
    P = np.array(r1+r2+r3)

    pi0 = np.array([1.,0.,0.])
    col1, col2, col3 = st.columns([1,1,1])
    with col1: steps = st.slider("스텝", 2, 120, 40, key=f"{PFX}:steps")
    with col2: fps   = st.slider("FPS", 2, 30, 12, key=f"{PFX}:fps")
    with col3: autorun = st.checkbox("자동 재생", True, key=f"{PFX}:auto")

    k = next_frame_index(PFX, steps+1, fps, autorun)
    pi = pi0 @ np.linalg.matrix_power(P, k)

    fig = go.Figure()
    fig.add_bar(x=[f"s{i+1}" for i in range(3)], y=pi, marker_color=["#e74c3c","#27ae60","#2980b9"])
    fig.update_layout(template="plotly_white", yaxis=dict(range=[0,1]), title=f"분포 π_k (k={k})")
    st.plotly_chart(fig, use_container_width=True)
