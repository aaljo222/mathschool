# tabs/tab_convolution.py
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from utils.anim import playbar, step_loop

PFX = "conv1d"

def render():
    st.subheader("1D 컨볼루션: 뒤집고 밀며 적분하기")
    st.latex(r"y(\tau)=(x*h)(\tau)=\int_{-\infty}^{\infty}x(t)\,h(t-\tau)\,dt")

    # 타임축
    L = 8.0
    t = np.linspace(-L, L, 1601)   # 홀수 길이가 인덱싱에 유리
    dt = t[1] - t[0]

    c1, c2, c3 = st.columns(3)
    with c1:
        a  = st.slider("x(t) 중심", -3.0, 3.0, -1.0, 0.1, key=f"{PFX}:a")
        sx = st.slider("x 폭",    0.2,  2.5, 0.6,  0.05, key=f"{PFX}:sx")
    with c2:
        b  = st.slider("h(t) 중심", -3.0, 3.0,  1.0, 0.1, key=f"{PFX}:b")
        sh = st.slider("h 폭",     0.2,  2.5, 0.8,  0.05, key=f"{PFX}:sh")
    with c3:
        fps  = st.slider("FPS",     2, 30, 12, key=f"{PFX}:fps")
        secs = st.slider("길이(초)", 1, 12,  6, key=f"{PFX}:secs")

    # 가우시안 예시 신호
    x = np.exp(-(t - a) ** 2 / (2 * sx ** 2))
    h = np.exp(-(t - b) ** 2 / (2 * sh ** 2))

    # 전체 컨볼루션(정적) : y_full(τ)
    y_full = np.convolve(x, h[::-1], mode="same") * dt

    # 🔁 애니메이션 컨트롤
    playing = playbar(PFX)
    steps   = max(2, int(secs * fps))

    # ⬇️ 루프에서 갱신할 출력은 placeholder로
    top_ph    = st.empty()   # x, h(t-τ), 곱과 적분영역
    bottom_ph = st.empty()   # y_full과 현재 τ 마커

    def draw(frac: float):
        # τ 이동 ([-L, L] 범위)
        tau = (2 * frac - 1) * L

        # h(t-τ) = h( (t) - τ ) = h( 뒤집고 + 평행이동 )
        # t' = t[::-1] - tau 에서 보간
        h_shift = np.interp(t, t[::-1] - tau, h, left=0.0, right=0.0)

        prod = x * h_shift
        val  = prod.sum() * dt   # y(τ) 근사치

        # ── 상단: 적분 그림 ─────────────────────────────────────
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=x,        mode="lines", name="x(t)"))
        fig.add_trace(go.Scatter(x=t, y=h_shift,  mode="lines", name="h(t-τ)"))
        fig.add_trace(go.Scatter(x=t, y=prod,     mode="lines", name="x·h(t-τ)", line=dict(width=1)))
        # 적분영역 살짝 색칠
        fig.add_trace(go.Scatter(x=t, y=prod, mode="lines", fill="tozeroy",
                                 name="적분영역", opacity=0.20, showlegend=False))
        fig.add_vline(x=tau, line_dash="dot")
        fig.update_layout(
            template="plotly_white", height=430,
            title=f"τ = {tau:.2f}   →   y(τ) ≈ {val:.3f}",
            xaxis_title="t", yaxis_title="amplitude"
        )
        top_ph.plotly_chart(fig, use_container_width=True)

        # ── 하단: 전체 컨볼루션 y(τ) ──────────────────────────
        # 현재 τ 위치의 y 표시
        idx = int(np.clip(np.searchsorted(t, tau), 0, len(t)-1))
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=t, y=y_full, mode="lines", name="(x*h)(τ)"))
        fig2.add_trace(go.Scatter(x=[tau], y=[y_full[idx]],
                                  mode="markers", name="현재 τ",
                                  marker=dict(size=9)))
        fig2.update_layout(
            template="plotly_white", height=260,
            xaxis_title="τ", yaxis_title="y(τ)"
        )
        bottom_ph.plotly_chart(fig2, use_container_width=True)

    if playing:
        for k in step_loop(steps, fps=fps, key=PFX):
            draw(k / (steps - 1))
    else:
        draw(0.5)   # 정지 상태에서는 가운데(τ≈0) 보여주기
