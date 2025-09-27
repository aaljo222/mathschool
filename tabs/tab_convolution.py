# tabs/tab_convolution.py
import numpy as np, streamlit as st, plotly.graph_objects as go
from utils.anim import playbar, step_loop

def render():
    st.subheader("이산 컨볼루션: 슬라이딩 곱-적분 시각화")
    x = np.array(list(map(float, st.text_input("x[n] (콤마)", "1,2,1").split(","))))
    h = np.array(list(map(float, st.text_input("h[n] (콤마)", "1,-1,1").split(","))))
    N = len(x); M = len(h); L = N+M-1
    n = np.arange(L)

    playing = playbar("conv")
    left, right = st.columns(2)
    phA = left.empty(); phB = right.empty()

    def draw(k):
        # k번째 출력 계산
        yk = 0.0
        xs = []; hs = []
        for m in range(N):
            j = k-m
            if 0 <= j < M:
                xs.append(m); hs.append(h[j])
                yk += x[m]*h[j]
        # 막대그래프
        fig1 = go.Figure()
        fig1.add_bar(x=np.arange(N), y=x, name="x[n]")
        fig1.add_bar(x=np.arange(M)+k-M+1, y=h[::-1], name="h[k-n]", opacity=0.5)
        fig1.update_layout(template="plotly_white", height=420)
        phA.plotly_chart(fig1, use_container_width=True)

        fig2 = go.Figure()
        yy = np.zeros(L); yy[k]=yk
        fig2.add_bar(x=n, y=yy, name="현재 y[k]", marker_color="#2c3e50")
        fig2.update_layout(template="plotly_white", height=420, xaxis_title="k", yaxis_title="y")
        phB.plotly_chart(fig2, use_container_width=True)

    if playing:
        for k in step_loop(L, fps=2, key="conv"):
            draw(k)
    else:
        draw(L//2)
