# tabs/tab_trig.py
import time
import numpy as np
import streamlit as st
from utils.plot import line_fig

PFX = "trig"

def _draw(f, A, phi):
    t = np.linspace(0, 2, 1000)
    y_sin = A * np.sin(2*np.pi*f*t + phi)
    y_cos = A * np.cos(2*np.pi*f*t + phi)
    return line_fig(t, [y_sin, y_cos],
                    ["A·sin(2πft+φ)", "A·cos(2πft+φ)"],
                    "삼각함수")

def render():
    st.subheader("삼각함수: sin, cos (주파수/위상 조절 · 한 번 재생)")

    c = st.columns(3)
    with c[0]: f   = st.slider("주파수 f (Hz)", 0.1, 5.0, 1.0, 0.1, key=f"{PFX}:f")
    with c[1]: A   = st.slider("진폭 A",       0.5, 3.0, 1.0, 0.1, key=f"{PFX}:A")
    with c[2]: phi = st.slider("위상 φ (rad)", -np.pi, np.pi, 0.0, 0.01, key=f"{PFX}:phi")

    c2 = st.columns(3)
    with c2[0]:
        anim_target = st.selectbox("재생할 항목", ["위상 φ", "주파수 f"], key=f"{PFX}:target")
    with c2[1]:
        fps   = st.slider("FPS", 2, 30, 15, key=f"{PFX}:fps")
    with c2[2]:
        steps = st.slider("프레임 수", 20, 240, 120, key=f"{PFX}:steps")

    ph = st.empty()
    ph.plotly_chart(_draw(f, A, phi), use_container_width=True)

    if st.button("🎬 한 번 재생", key=f"{PFX}:once"):
        st.session_state[f"{PFX}:playing"] = True

    # 버튼 누른 경우에만 1회 재생 (next_frame_index 사용 안 함)
    if st.session_state.get(f"{PFX}:playing", False):
        for k in range(steps):
            if anim_target == "위상 φ":
                phi_k = -np.pi + 2*np.pi * k / max(1, steps-1)
                fig = _draw(f, A, phi_k)
            else:  # 주파수 f
                f_k = 0.1 + (5.0 - 0.1) * k / max(1, steps-1)
                fig = _draw(f_k, A, phi)

            ph.plotly_chart(fig, use_container_width=True)
            time.sleep(1.0 / max(1, fps))

        st.session_state[f"{PFX}:playing"] = False
