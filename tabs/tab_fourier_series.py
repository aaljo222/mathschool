# tabs/tab_fourier_series.py
import numpy as np
import streamlit as st
import plotly.graph_objects as go

PFX = "fsq"

def render():
    st.subheader("푸리에 급수: 사각파 근사 (이벤트 기반 1회 렌더)")

    # 표시할 항 개수(홀수만)
    Nmax = st.slider("최대 항 수(홀수 권장)", 1, 201, 31, step=1, key=f"{PFX}:Nmax")
    N    = st.slider("표시할 항 N (홀수)", 1, Nmax, min(Nmax, 31), step=1, key=f"{PFX}:N")
    if N % 2 == 0:
        N -= 1
        st.info(f"N은 홀수만 사용합니다 → N={N}")

    show_target = st.checkbox("이상적 사각파(참값) 겹쳐 보기", True, key=f"{PFX}:target")

    # 부분합 계산 (사각파: 홀수항만)
    x = np.linspace(-np.pi, np.pi, 1600)
    y = np.zeros_like(x)
    for n in range(1, N+1, 2):  # 홀수항
        y += (4/np.pi) * (1/n) * np.sin(n * x)

    fig = go.Figure()
    fig.add_scatter(x=x, y=y, mode="lines", name=f"부분합 S_N (N={N})", line=dict(width=3))

    if show_target:
        target = np.sign(np.sin(x))  # 진폭 ±1 사각파
        fig.add_scatter(x=x, y=target, mode="lines", name="사각파(참값)",
                        line=dict(width=1, dash="dot"))

    fig.update_layout(
        template="plotly_white",
        height=480,
        xaxis_title="x", yaxis_title="y",
        title="사각파 푸리에 부분합  S_N(x) = Σ_{n=1,3,5,...,N} (4/π)(1/n) sin(nx)"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption("N을 크게 해도 불연속점 근처의 약 9% 오버슈트(Gibbs 현상)는 사라지지 않습니다.")
