# tabs/tab_markov.py
import numpy as np, streamlit as st, plotly.graph_objects as go
from utils.anim import playbar, step_loop

def render():
    st.subheader("3-상태 마르코프 체인: pₜ₊₁ = pₜ P")
    cols = st.columns(3)
    with cols[0]:
        p = np.array([st.number_input("p₀(A)", 0.7), st.number_input("p₀(B)", 0.2), st.number_input("p₀(C)", 0.1)], float)
        p = p/np.sum(p)
    with cols[1]:
        A = st.slider("A→(A,B,C)", 0.0, 1.0, (0.6, 0.3))
        PA = np.array([A[0], A[1]-A[0], 1-A[1]])
    with cols[2]:
        B = st.slider("B→(A,B,C)", 0.0, 1.0, (0.2, 0.7))
        PB = np.array([B[0], B[1]-B[0], 1-B[1]])
    C1 = st.slider("C→A", 0.0, 1.0, 0.1); C2 = st.slider("C→A+B", 0.0, 1.0, 0.5)
    PC = np.array([C1, max(C2-C1,0.0), max(1-C2,0.0)])
    P = np.vstack([PA, PB, PC])  # rows sum 1

    steps = st.slider("스텝 수", 1, 60, 25); fps = st.slider("FPS", 2, 20, 8)
    playing = playbar("markov"); ph = st.empty()

    def draw(t):
        pt = p.copy()
        hist = [pt]
        for _ in range(t):
            pt = pt @ P
            hist.append(pt)
        H = np.array(hist)
        fig = go.Figure()
        for i, name in enumerate(["A","B","C"]):
            fig.add_trace(go.Scatter(y=H[:,i], x=list(range(len(H))), mode="lines+markers", name=name))
        fig.update_layout(template="plotly_white", height=480, xaxis_title="t", yaxis_title="probability", yaxis=dict(range=[0,1]))
        ph.plotly_chart(fig, use_container_width=True)

    if playing:
        for k in step_loop(steps, fps=fps, key="markov"):
            draw(k)
    else:
        draw(steps)
