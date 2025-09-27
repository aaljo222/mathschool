# tabs/tab_markov.py
import numpy as np, streamlit as st, plotly.graph_objects as go
from utils.anim import playbar, step_loop

PFX = "markov"  # 이 탭만의 고유 접두사

def render():
    st.subheader("마르코프 체인: 분포의 시간 전개")

    # 예시 전이행렬(3상태) 입력
    c = st.columns(3)
    r1 = [c[0].slider("p11", 0.0, 1.0, 0.7, 0.01, key=f"{PFX}:p11"),
          c[1].slider("p12", 0.0, 1.0, 0.2, 0.01, key=f"{PFX}:p12"),
          c[2].slider("p13", 0.0, 1.0, 0.1, 0.01, key=f"{PFX}:p13")]
    c = st.columns(3)
    r2 = [c[0].slider("p21", 0.0, 1.0, 0.1, 0.01, key=f"{PFX}:p21"),
          c[1].slider("p22", 0.0, 1.0, 0.8, 0.01, key=f"{PFX}:p22"),
          c[2].slider("p23", 0.0, 1.0, 0.1, 0.01, key=f"{PFX}:p23")]
    c = st.columns(3)
    r3 = [c[0].slider("p31", 0.0, 1.0, 0.2, 0.01, key=f"{PFX}:p31"),
          c[1].slider("p32", 0.0, 1.0, 0.2, 0.01, key=f"{PFX}:p32"),
          c[2].slider("p33", 0.0, 1.0, 0.6, 0.01, key=f"{PFX}:p33")]

    def _norm(row):
        s = sum(row) or 1.0
        return [x/s for x in row]

    P = np.array([_norm(r1), _norm(r2), _norm(r3)])
    pi0 = np.array([1.0, 0.0, 0.0])  # 초기분포

    left, right = st.columns(2)
    with left:
        steps = st.slider("스텝 수", 2, 60, 20, key=f"{PFX}:steps")
    with right:
        fps   = st.slider("FPS", 2, 30, 12, key=f"{PFX}:fps")

    playing = playbar(PFX)  # ▶ / ⏸ 버튼 (키는 PFX로)

    ph = st.empty()  # ← 반드시 1개의 placeholder만 사용

    def draw(k: int):
        dist = pi0 @ np.linalg.matrix_power(P, k)
        fig = go.Figure()
        fig.add_bar(x=[f"s{i+1}" for i in range(3)], y=dist, name=f"k={k}")
        fig.update_layout(template="plotly_white",
                          yaxis=dict(range=[0, 1]),
                          title=f"분포 π_k (k={k})")
        ph.plotly_chart(fig, use_container_width=True)

    if playing:
        # 같은 run 안에서 같은 자리의 차트를 중복 생성하지 않도록
        for k in step_loop(steps + 1, fps=fps, key=PFX):
            ph.empty()     # ← 먼저 비우고
            draw(k)        # ← 한 번만 그린다
    else:
        ph.empty()
        draw(steps)        # 일시정지일 때는 1회만 그림
