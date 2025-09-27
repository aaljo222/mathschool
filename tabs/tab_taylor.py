# tabs/tab_taylor.py
import time
import math
import numpy as np
import streamlit as st
from utils.plot import line_fig

PFX = "taylor"

# --------- 수식 유틸 ----------
def taylor_series(kind: str, X: np.ndarray, c: float, N: int) -> np.ndarray:
    """f의 c에서의 N차 테일러 다항식 값을 X에서 계산"""
    T = np.zeros_like(X)
    if kind == "exp":
        # f^(n)(c) = e^c
        ec = math.exp(c)
        for n in range(N + 1):
            T += (ec / math.factorial(n)) * (X - c) ** n
    elif kind == "sin":
        derivs = [np.sin, np.cos, lambda z: -np.sin(z), lambda z: -np.cos(z)]
        for n in range(N + 1):
            T += (derivs[n % 4](c) / math.factorial(n)) * (X - c) ** n
    elif kind == "cos":
        derivs = [np.cos, lambda z: -np.sin(z), lambda z: -np.cos(z), np.sin]
        for n in range(N + 1):
            T += (derivs[n % 4](c) / math.factorial(n)) * (X - c) ** n
    return T

def _draw(kind, f_true, X, c, N, subtitle: str = ""):
    T = taylor_series(kind, X, c, N)
    title = f"Taylor around c={c:.2f}  ·  N={N}"
    if subtitle:
        title += f"   ({subtitle})"
    return line_fig(X, [f_true(X), T], ["원함수", f"테일러 근사 (N={N})"], title)

# --------- UI / 렌더 ----------
def render():
    st.subheader("테일러 시리즈 근사 (한 번 재생)")

    funcs = {
        "sin x": (lambda x: np.sin(x), "sin"),
        "cos x": (lambda x: np.cos(x), "cos"),
        "e^x"  : (lambda x: np.exp(x), "exp"),
    }

    c0, c1 = st.columns([1.2, 1])
    with c0:
        fname = st.selectbox("함수 선택", list(funcs.keys()), key=f"{PFX}:fn")
    with c1:
        Xmin, Xmax = st.slider("x-범위", -10.0, 10.0, (-6.0, 6.0), 0.5, key=f"{PFX}:xrange")

    f_true, kind = funcs[fname]
    X = np.linspace(Xmin, Xmax, 1200)

    a, b, c = st.columns(3)
    with a:
        c_center = st.slider("전개 중심 c", -4.0, 4.0, 0.0, 0.1, key=f"{PFX}:c")
    with b:
        N_order  = st.slider("차수 N", 0, 20, 6, 1, key=f"{PFX}:N")
    with c:
        anim_target = st.selectbox("재생 대상", ["차수 N", "중심 c"], key=f"{PFX}:target")

    d, e, f = st.columns(3)
    with d:
        fps = st.slider("FPS", 2, 30, 15, key=f"{PFX}:fps")
    with e:
        steps = st.slider("프레임 수", 10, 200, 80, key=f"{PFX}:steps")
    with f:
        c_range = st.slider("c 애니메이션 범위", -4.0, 4.0, (-2.0, 2.0), 0.1, key=f"{PFX}:crange")

    # 정적 미리보기
    holder = st.empty()
    holder.plotly_chart(_draw(kind, f_true, X, c_center, N_order), use_container_width=True)

    if st.button("🎬 한 번 재생", key=f"{PFX}:once"):
        st.session_state[f"{PFX}:playing"] = True

    # 버튼을 눌렀을 때만 1회 애니메이션
    if st.session_state.get(f"{PFX}:playing", False):
        if anim_target == "차수 N":
            for n in range(N_order + 1):
                fig = _draw(kind, f_true, X, c_center, n, "N 증가")
                holder.plotly_chart(fig, use_container_width=True)
                time.sleep(1.0 / max(1, fps))
        else:  # 중심 c
            c1, c2 = c_range
            for k in range(steps):
                cc = c1 + (c2 - c1) * k / max(1, steps - 1)
                fig = _draw(kind, f_true, X, cc, N_order, "c 이동")
                holder.plotly_chart(fig, use_container_width=True)
                time.sleep(1.0 / max(1, fps))

        st.session_state[f"{PFX}:playing"] = False
