# tabs/tab_convolution.py
import time
import numpy as np
import streamlit as st
import plotly.graph_objects as go

PFX = "conv1d"

def render():
    st.subheader("1D 컨볼루션: 뒤집고 밀며 적분하기")

    L = 8.0
    t = np.linspace(-L, L, 1200)

    c1, c2, c3 = st.columns(3)
    with c1:
        a  = st.slider("x(t) 중심 a", -3.0, 3.0, -1.0, 0.1, key=f"{PFX}:a")
        sx = st.slider("x 폭 σx", 0.2, 2.5, 0.6, 0.05, key=f"{PFX}:sx")
    with c2:
        b  = st.slider("h(t) 중심 b", -3.0, 3.0,  1.0, 0.1, key=f"{PFX}:b")
        sh = st.slider("h 폭 σh", 0.2, 2.5, 0.8, 0.05, key=f"{PFX}:sh")
    with c3:
        secs = st.slider("애니메이션 길이(초)", 1, 12, 6, key=f"{PFX}:secs")
        fps  = st.slider("FPS", 5, 40, 20, key=f"{PFX}:fps")

    # 신호 정의
    x = np.exp(-(t - a)**2 / (2*sx**2))
    h = np.exp(-(t - b)**2 / (2*sh**2))
    dt = t[1] - t[0]

    # 전체 컨볼루션(정답 곡선) 미리 계산
    y_full = np.convolve(x, h[::-1], mode="same") * dt

    # 출력 자리
    ph_main = st.empty()
    ph_conv = st.empty()

    # 정지 화면용 τ 슬라이더 + 한 번 재생 버튼
    colA, colB = st.columns([2, 1])
    with colA:
        tau_manual = st.slider("정지 화면 τ", -L, L, 0.0, 0.05, key=f"{PFX}:tau")
    with colB:
        run_once = st.button("▶ 한 번 재생 (τ: −L → +L)", use_container_width=True)

    def draw_at_tau(tau: float):
        # h(−τ)은 h(t)를 뒤집고 τ만큼 이동한 것과 동일: h(tau − t)
        h_flip_shift = np.interp(t, t[::-1] - tau, h, left=0.0, right=0.0)
        prod = x * h_flip_shift
        val  = float(prod.sum() * dt)

        fig = go.Figure()
        fig.add_scatter(x=t, y=x,              mode="lines", name="x(t)")
        fig.add_scatter(x=t, y=h_flip_shift,   mode="lines", name="h(−τ)")
        fig.add_scatter(x=t, y=prod,           mode="lines", name="곱 x·h(−τ)", line=dict(width=1))
        fig.add_vline(x=tau, line_dash="dot")
        fig.update_layout(template="plotly_white", height=430,
                          title=f"τ = {tau:.2f},   y(τ) ≈ {val:.4f}")
        ph_main.plotly_chart(fig, use_container_width=True)

        # 전체 y(t)와 현재 τ 지점 표시
        fig2 = go.Figure()
        fig2.add_scatter(x=t, y=y_full, mode="lines", name="y(t) = x ∗ h")
        # 현재 τ 위치의 지점
        fig2.add_vline(x=tau, line_dash="dot")
        # 근사적으로 현재값 마커 (보간)
        y_now = float(np.interp(tau, t, y_full))
        fig2.add_scatter(x=[tau], y=[y_now], mode="markers", name="y(τ)", marker=dict(size=9))
        fig2.update_layout(template="plotly_white", height=260, title=f"컨볼루션 결과곡선 · y(τ)≈{y_now:.4f}")
        ph_conv.plotly_chart(fig2, use_container_width=True)

    if not run_once:
        # 정지화면 렌더
        draw_at_tau(tau_manual)
        return

    # 버튼 클릭 시: 한 번만 −L → +L 스윕
    total_frames = max(2, int(secs * fps))
    taus = np.linspace(-L, L, total_frames)
    start = time.perf_counter()
    for i, tau in enumerate(taus, 1):
        draw_at_tau(float(tau))
        # 일정한 FPS 유지
        target = i / fps
        sleep = target - (time.perf_counter() - start)
        if sleep > 0:
            time.sleep(sleep)
