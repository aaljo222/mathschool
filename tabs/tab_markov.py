import numpy as np, streamlit as st, plotly.graph_objects as go
from utils.anim import playbar, step_loop

PFX = "markov3"

def render():
    st.subheader("3상태 마르코프 체인의 분포 수렴")

    col1, col2 = st.columns(2)
    with col1:
        alpha = st.slider("머무를 확률 α", 0.0, 1.0, 0.5, 0.01, key=f"{PFX}:alpha")
        beta  = st.slider("시작 분포 b=(b1,b2,b3)", 0.0, 1.0, 0.33, 0.01, key=f"{PFX}:b1")
        b     = np.array([beta, (1-beta)/2, (1-beta)/2])
    with col2:
        secs = st.slider("길이(초)", 1, 12, 6, key=f"{PFX}:secs")
        fps  = st.slider("FPS", 2, 30, 12, key=f"{PFX}:fps")

    # i→i: α, i→(i+1 mod 3): 1-α
    T = np.array([[alpha, 1-alpha, 0],
                  [0,     alpha,  1-alpha],
                  [1-alpha, 0,    alpha]], float)

    playing = playbar(PFX)

    # ⬇️ 루프에서 갱신하는 모든 출력은 placeholder 사용
    chart_ph = st.empty()
    text_ph  = st.empty()

    steps   = max(2, int(secs*fps))

    def draw(t):
        k = int(t*(steps-1))
        p = b @ np.linalg.matrix_power(T, k)
        fig = go.Figure(go.Bar(x=["A","B","C"], y=p,
                               marker_color=["#e67e22","#27ae60","#2980b9"]))
        fig.update_yaxes(range=[0,1])
        fig.update_layout(template="plotly_white", height=420,
                          title=f"분포 p_k (k={k})", bargap=0.25)
        chart_ph.plotly_chart(fig, use_container_width=True)

        text_ph.caption("T = " + str(T.round(3).tolist()))

    if playing:
        for i in step_loop(steps, fps=fps, key=PFX):
            draw(i/(steps-1))
    else:
        draw(0.0)
