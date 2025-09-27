# tabs/tab_log_add.py
import time
import numpy as np
import streamlit as st
import plotly.graph_objects as go

PFX = "logmul"

def render():
    st.subheader("곱셈 ↔ 로그-덧셈 ( log(ab) = log a + log b ) — 버튼을 누를 때만 1회 재생")

    # ── 로그 밑 선택 ──────────────────────────────────────────────────────
    base_sel = st.selectbox("로그 밑", ["e (자연로그)", "10"], index=0, key=f"{PFX}:base")
    if base_sel.startswith("e"):
        log_fn = np.log
        exp_fn = np.exp
        log_label = "ln";  exp_label = "e^{x}"
    else:
        log_fn = np.log10
        exp_fn = lambda x: 10**x
        log_label = "log_{10}";  exp_label = "10^{x}"

    # ── 파라미터(양수 유지) ───────────────────────────────────────────────
    c = st.columns(4)
    with c[0]: a0 = st.slider("a 오프셋", 0.1, 5.0, 1.2, 0.1, key=f"{PFX}:a0")
    with c[1]: aA = st.slider("a 진폭",   0.0, 3.0, 1.0, 0.1, key=f"{PFX}:aA")
    with c[2]: b0 = st.slider("b 오프셋", 0.1, 5.0, 1.0, 0.1, key=f"{PFX}:b0")
    with c[3]: bA = st.slider("b 진폭",   0.0, 3.0, 0.8, 0.1, key=f"{PFX}:bA")

    c2 = st.columns(3)
    with c2[0]: w1    = st.slider("a 주기(클수록 느림)", 1, 12, 6,  key=f"{PFX}:w1")
    with c2[1]: w2    = st.slider("b 주기",             1, 12, 8,  key=f"{PFX}:w2")
    with c2[2]: phase = st.slider("위상차 φ (rad)",     0.0, 2*np.pi, 1.0, 0.01, key=f"{PFX}:phi")

    # 애니메이션 길이(최대 5초 내외 권장)
    fps   = st.slider("FPS", 2, 30, 12, key=f"{PFX}:fps")
    steps = st.slider("프레임 수(1회 재생)", 20, 240, 120, key=f"{PFX}:steps")

    # ── 출력 자리(항상 같은 자리에만 그리기) ─────────────────────────────
    ph1 = st.empty()       # 곱셈 막대 그래프
    ph2 = st.empty()       # 로그-덧셈 막대 그래프
    info = st.empty()      # 공식을 캡션으로 갱신

    # 진행률 슬라이더(수동 스크럽)
    frac = st.slider("진행률(수동)", 0.0, 1.0, 0.0, 0.01, key=f"{PFX}:scrub")

    # 1프레임을 그리는 함수(두 그래프를 같은 자리에 갱신)
    def draw(frac: float):
        tau = 2*np.pi * frac
        a = a0 + aA*(1 + np.sin(tau*w1))
        b = b0 + bA*(1 + np.sin(tau*w2 + phase))
        ab = a * b

        la, lb = log_fn(a), log_fn(b)
        sum_logs = la + lb
        log_ab   = log_fn(ab)
        exp_sum  = exp_fn(sum_logs)
        err      = float(exp_sum - ab)

        # (1) 곱셈 영역
        fig1 = go.Figure()
        fig1.add_bar(x=["a", "b", "a·b"], y=[a, b, ab],
                     marker_color=["#1f77b4", "#2ca02c", "#d62728"])
        fig1.update_layout(template="plotly_white", height=360, yaxis_title="값",
                           title=f"a={a:.3f},  b={b:.3f},  a·b={ab:.3f}")
        ph1.plotly_chart(fig1, use_container_width=True)

        # (2) 로그-덧셈 영역
        fig2 = go.Figure()
        fig2.add_bar(x=[f"{log_label} a", f"{log_label} b", f"{log_label}(a·b)"],
                     y=[la, lb, log_ab],
                     marker_color=["#1f77b4", "#2ca02c", "#9467bd"])
        fig2.add_scatter(x=["합"], y=[sum_logs], mode="markers",
                         marker=dict(size=14, symbol="diamond", color="#ff7f0e"),
                         name=f"합({log_label}a+{log_label}b)")
        fig2.update_layout(template="plotly_white", height=360, yaxis_title="로그 값",
                           title=f"{log_label}(a·b)  =  {log_label}a + {log_label}b  ≈  {sum_logs:.3f}")
        ph2.plotly_chart(fig2, use_container_width=True)

        with info.container():
            st.latex(rf"{log_label}(ab) = {log_label}a + {log_label}b,\qquad {exp_label}({log_label}a + {log_label}b) = ab")
            st.caption(f"역로그 복원: exp(sum) = {exp_sum:.6f},  a·b = {ab:.6f},  오차 = {err:.2e} (수치오차) ")

    # 기본 화면(정지 프레임)
    draw(frac)

    # ── ‘한 번 재생’ 버튼 (누를 때만 1회 루프) ─────────────────────────────
    if st.button("🎬 한 번 재생", key=f"{PFX}:once"):
        st.session_state[f"{PFX}:playing"] = True

    if st.session_state.get(f"{PFX}:playing", False):
        for k in range(steps):
            draw(k / max(1, steps-1))
            time.sleep(1.0 / max(1, fps))
        st.session_state[f"{PFX}:playing"] = False
