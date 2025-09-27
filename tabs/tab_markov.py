# tabs/tab_newton.py
import time
import numpy as np
import streamlit as st
import plotly.graph_objects as go

PFX = "newton"

# 간단한 함수 세트 (필요하면 추가)
FUNCS = {
    "f(x)=x^3 - x - 1": lambda x: x**3 - x - 1,
    "f(x)=cos x - x": lambda x: np.cos(x) - x,
    "f(x)=x^2 - 2": lambda x: x**2 - 2,
}

def deriv(f, x, h=1e-6):
    # 수치 미분
    return (f(x + h) - f(x - h)) / (2*h)

def _newton_path(f, x0, nmax, safe_eps=1e-12):
    xs = [x0]
    for _ in range(nmax):
        xk = xs[-1]
        d = deriv(f, xk)
        if abs(d) < safe_eps:  # 기울기 너무 작으면 중단
            break
        xnext = xk - f(xk)/d
        xs.append(xnext)
        if not np.isfinite(xnext):
            break
    return np.array(xs)

def _draw(f, xs, k, x_range):
    # k번째까지의 뉴턴 경로를 그림
    xgrid = np.linspace(*x_range, 400)
    ygrid = f(xgrid)

    fig = go.Figure()
    # 함수 그래프
    fig.add_scatter(x=xgrid, y=ygrid, mode="lines", name="f(x)")

    # x축
    fig.add_hline(y=0, line_color="#aaa", line_dash="dot")

    # 점/접선
    k = int(np.clip(k, 0, len(xs)-1))
    for i in range(0, k):
        xk = xs[i]
        yk = f(xk)
        dk = deriv(f, xk)
        # 접선 y = f(xk) + dk*(x-xk)
        xline = np.array([xk - 1.0, xk + 1.0])
        yline = yk + dk*(xline - xk)
        fig.add_scatter(x=xline, y=yline, mode="lines",
                        line=dict(width=2, dash="dash"),
                        name=f"tangent@k={i}")

        # 수직선 x=x_{k+1}
        if i+1 < len(xs):
            xnext = xs[i+1]
            fig.add_vline(x=xnext, line_color="#888", line_dash="dot")

    # 현재 점 마커
    xk = xs[k]
    fig.add_scatter(x=[xk], y=[f(xk)], mode="markers",
                    marker=dict(size=10, color="#e74c3c"), name=f"x{k}")

    # 표시 범위와 레이아웃
    ymin, ymax = np.percentile(ygrid, [5, 95])
    pad = 0.2*(ymax - ymin + 1e-9)
    fig.update_layout(template="plotly_white",
                      height=520,
                      xaxis=dict(range=x_range, zeroline=False),
                      yaxis=dict(range=[ymin-pad, ymax+pad], zeroline=True),
                      title=f"Newton's method (k={k}, x_k ≈ {xk:.6g})")
    return fig

def render():
    st.subheader("뉴턴법: 한 번 재생(Play once) / 수동 스크럽")

    # 함수 선택 + 범위
    c0, c1 = st.columns([1, 1])
    with c0:
        fname = st.selectbox("함수 선택", list(FUNCS.keys()), index=0, key=f"{PFX}:fsel")
    with c1:
        x_min, x_max = st.slider("x-표시 범위", -5.0, 5.0, (-2.0, 2.0), 0.1, key=f"{PFX}:xrange")
    f = FUNCS[fname]

    # 시작점 / 스텝 / FPS
    c2, c3, c4 = st.columns(3)
    with c2:
        x0 = st.slider("초기값 x0", x_min, x_max, 1.5, 0.01, key=f"{PFX}:x0")
    with c3:
        nmax = st.slider("최대 스텝", 1, 50, 12, key=f"{PFX}:nmax")
    with c4:
        fps = st.slider("FPS", 2, 30, 12, key=f"{PFX}:fps")

    # 경로 미리 계산
    xs = _newton_path(f, x0, nmax)
    total = len(xs) - 1  # 이동 단계 수(k=0..total)

    # 고정 플레이스홀더
    ph_chart = st.empty()
    ph_info = st.empty()

    # 수동 스크럽
    k_scrub = st.slider("현재 k(수동)", 0, total, 0, key=f"{PFX}:k")

    # 정지화면 먼저 렌더
    ph_chart.plotly_chart(_draw(f, xs, k_scrub, (x_min, x_max)), use_container_width=True)
    ph_info.info(f"현재 k={k_scrub}, x_k≈{xs[k_scrub]:.6g}, f(x_k)≈{f(xs[k_scrub]):.3e}")

    # 한 번 재생
    if st.button("🎬 한 번 재생", key=f"{PFX}:once"):
        st.session_state[f"{PFX}:playing"] = True

    if st.session_state.get(f"{PFX}:playing", False):
        for k in range(total + 1):
            ph_chart.plotly_chart(_draw(f, xs, k, (x_min, x_max)), use_container_width=True)
            ph_info.info(f"현재 k={k}, x_k≈{xs[k]:.6g}, f(x_k)≈{f(xs[k]):.3e}")
            time.sleep(1.0 / max(1, fps))
        st.session_state[f"{PFX}:playing"] = False
