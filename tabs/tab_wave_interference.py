# tabs/tab_wave_interference.py
import time
import numpy as np
import streamlit as st
import plotly.graph_objects as go

PFX = "wave2"

def _draw(f1, f2, A, phase):
    """현재 phase(=t0)에 대한 파형들을 그립니다."""
    x = np.linspace(0, 2*np.pi, 800)
    y1 = A * np.sin(f1 * x - phase)
    y2 = A * np.sin(f2 * x - phase)
    y  = y1 + y2

    fig = go.Figure()
    fig.add_scatter(x=x, y=y1, mode="lines", name="y₁")
    fig.add_scatter(x=x, y=y2, mode="lines", name="y₂")
    fig.add_scatter(x=x, y=y,  mode="lines", name="합성", line=dict(width=3))
    fig.update_layout(
        template="plotly_white",
        height=480,
        title=f"두 사인파 간섭 (phase t₀ = {phase:.2f} rad)"
    )
    return fig

def render():
    st.subheader("두 사인파 간섭 (Play once / 수동 스크럽)")

    c = st.columns(3)
    with c[0]: f1 = st.slider("f₁ (Hz)", 0.5, 6.0, 2.0, 0.1, key=f"{PFX}:f1")
    with c[1]: f2 = st.slider("f₂ (Hz)", 0.5, 6.0, 2.2, 0.1, key=f"{PFX}:f2")
    with c[2]: A  = st.slider("진폭 A",   0.2, 2.0, 1.0, 0.1, key=f"{PFX}:A")

    c2 = st.columns(2)
    with c2[0]: fps   = st.slider("FPS", 2, 30, 15, key=f"{PFX}:fps")
    with c2[1]: steps = st.slider("프레임 수", 20, 240, 120, key=f"{PFX}:steps")

    # 수동 스크럽(정지 화면)
    phase_manual = st.slider("수동 phase t₀ (rad)", 0.0, float(2*np.pi), 0.0, 0.01, key=f"{PFX}:phase")

    ph = st.empty()
    ph.plotly_chart(_draw(f1, f2, A, phase_manual), use_container_width=True)

    # 1회 재생 버튼
    if st.button("🎬 한 번 재생", key=f"{PFX}:once"):
        st.session_state[f"{PFX}:playing"] = True

    # 버튼이 눌리면 steps만큼 한 번만 애니메이션
    if st.session_state.get(f"{PFX}:playing", False):
        for k in range(steps):
            phase = 2*np.pi * k / max(1, steps-1)
            ph.plotly_chart(_draw(f1, f2, A, phase), use_container_width=True)
            time.sleep(1.0 / max(1, fps))
        st.session_state[f"{PFX}:playing"] = False
