# tabs/tab_exp_log.py
import time
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# ---------- 내부 헬퍼 ----------
def _ln(x):  # 안전한 ln
    x = np.clip(x, 1e-9, None)
    return np.log(x)

def _log_a(x, a):
    return _ln(x) / np.log(a)

def _make_fig(a, x_now, view_L, n=400):
    # 곡선 샘플
    xs = np.linspace(-view_L, view_L, n)
    y_exp = np.power(a, xs)

    # 로그 곡선은 x>0 에서만
    x_log = np.linspace(1e-3, np.power(a, view_L), n)
    y_log = _log_a(x_log, a)

    # 현재 프레임의 대응점
    y_now = np.power(a, x_now)          # P = (x_now, a^x_now)
    xr, yr = y_now, x_now               # Q = (a^x_now, x_now)  (y=x 대칭)

    fig = go.Figure()

    # y = a^x
    fig.add_trace(go.Scatter(
        x=xs, y=y_exp, mode="lines", name="y = a^x",
        line=dict(width=2)
    ))
    # y = log_a x
    fig.add_trace(go.Scatter(
        x=x_log, y=y_log, mode="lines", name="y = logₐ x",
        line=dict(width=2, dash="dash")
    ))
    # 대칭선 y=x
    fig.add_trace(go.Scatter(
        x=[-view_L, view_L], y=[-view_L, view_L],
        mode="lines", name="y = x", line=dict(width=1, dash="dot"),
        showlegend=True
    ))

    # 현재 위치: P(x, a^x) 와 Q(a^x, x)
    fig.add_trace(go.Scatter(
        x=[x_now], y=[y_now], mode="markers+text",
        marker=dict(size=9), name="P: (x, a^x)",
        text=[f"P({x_now:.2f}, {y_now:.2f})"], textposition="top center", showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=[xr], y=[yr], mode="markers+text",
        marker=dict(size=9), name="Q: (a^x, x)",
        text=[f"Q({xr:.2f}, {yr:.2f})"], textposition="bottom right", showlegend=False
    ))
    # P-Q 연결(대칭 대응)
    fig.add_trace(go.Scatter(
        x=[x_now, xr], y=[y_now, yr], mode="lines",
        line=dict(width=1, dash="dot", color="#7f8c8d"),
        name="reflection", showlegend=False
    ))

    fig.update_layout(
        template="plotly_white",
        height=520,
        margin=dict(l=10,r=10,t=40,b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        title=f"역함수 쌍대:  y = a^x  ↔  y = logₐ x   (a = {a:.2f})"
    )
    # y=x 대칭이 잘 보이도록 축 범위/비율 고정
    fig.update_xaxes(range=[-view_L, view_L])
    fig.update_yaxes(range=[-view_L, view_L], scaleanchor="x", scaleratio=1)

    return fig


# ---------- 탭 엔트리 ----------
def render():
    st.subheader("지수와 로그의 쌍대(역함수) 애니메이션")

    c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.2, 1.6])
    with c1:
        a = st.slider("밑 a (>0, ≠1)", 0.2, 5.0, 2.0, 0.01,
                      help="1에 너무 가까우면 그래프가 평평해 보여요.")
        # a=1 회피
        if 0.99 < a < 1.01:
            a = 1.01
    with c2:
        view_L = st.slider("보기 범위 L", 1.5, 6.0, 3.0, 0.5,
                           help="축을 [-L, L]로 고정 (y=x 대칭을 뚜렷하게)")
    with c3:
        secs = st.slider("길이(초)", 2, 12, 6)
    with c4:
        fps  = st.slider("FPS", 10, 40, 24)

    st.latex(r"""
    \textbf{핵심 관계}\quad
    a^{\log_a x} = x,\qquad \log_a(a^x)=x,\qquad
    \log_a x = \frac{\ln x}{\ln a}\ (a>0,\ a\neq 1).
    """)

    # 수동 포인터 + 재생 컨트롤
    c5, c6 = st.columns([1,1])
    with c5:
        x_now = st.slider("x 포인터", -view_L, view_L, 0.0, 0.05)
    with c6:
        play_col1, play_col2 = st.columns(2)
        with play_col1:
            if st.button("▶ 재생", use_container_width=True, key="exp_log_play_btn"):
                st.session_state.setdefault("exp_log_play", False)
                st.session_state.exp_log_play = True
        with play_col2:
            if st.button("⏸ 정지", use_container_width=True, key="exp_log_stop_btn"):
                st.session_state.exp_log_play = False

    ph = st.empty()

    # 정지 시: 현재 슬라이더 값으로 그림 한 장
    if not st.session_state.get("exp_log_play", False):
        ph.plotly_chart(_make_fig(a, x_now, view_L), use_container_width=True)
        return

    # 재생: x가 -L → +L로 이동
    total_frames = int(secs * fps)
    xs = np.linspace(-view_L, view_L, total_frames)
    start = time.perf_counter()
    for i, xv in enumerate(xs, 1):
        if not st.session_state.get("exp_log_play", False):
            break
        ph.plotly_chart(_make_fig(a, float(xv), view_L), use_container_width=True)
        # 타임라인 맞추기
        target = i / fps
        sleep = target - (time.perf_counter() - start)
        if sleep > 0:
            time.sleep(sleep)
    st.session_state.exp_log_play = False
