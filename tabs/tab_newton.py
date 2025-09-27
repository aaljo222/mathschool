# tabs/tab_newton.py
import time
import numpy as np
import streamlit as st
import plotly.graph_objects as go

PFX = "newton1d"

# f, f' 정의 (원하면 쉽게 바꿀 수 있도록 함수로 둠)
def f(x):  return x**3 - x - 1
def fp(x): return 3*x**2 - 1

def _newton_iters(x0: float, steps: int, deriv_eps=1e-10):
    xs = [float(x0)]
    for _ in range(steps):
        xn  = xs[-1]
        dfx = fp(xn)
        if abs(dfx) < deriv_eps:
            break  # 접선 기울기 거의 0 → 중단
        xs.append(xn - f(xn)/dfx)
    return np.array(xs)

def _draw_frame(xline, xs, k, xr=(-2.2, 2.2), yr=(-3, 3)):
    xn  = xs[k]
    fxn = f(xn)
    dfx = fp(xn) if abs(fp(xn)) > 1e-12 else (np.sign(fp(xn)) * 1e-12 or 1e-12)
    xnext = xn - fxn/dfx

    fig = go.Figure()
    # f(x)
    fig.add_scatter(x=xline, y=f(xline), mode="lines", name="f(x)")
    fig.add_hline(y=0, line_color="#888")
    # 현재 점 (xn, f(xn))
    fig.add_scatter(x=[xn], y=[fxn], mode="markers", name=f"x{k}", marker=dict(size=9))
    # 접선: y = f(xn) + f'(xn)(x - xn)
    ytan = fxn + dfx*(xline - xn)
    fig.add_scatter(x=xline, y=ytan, mode="lines", name="tangent", line=dict(dash="dot"))
    # 뉴턴 스텝 시각화: (xn, f(xn)) → (x_{n+1}, 0)
    fig.add_scatter(x=[xn, xnext], y=[fxn, 0], mode="lines",
                    name="Newton step", line=dict(width=3))
    fig.add_scatter(x=[xnext], y=[0], mode="markers",
                    name=f"x{k+1}", marker=dict(symbol="x", size=8))
    # 축 위 히스토리(이전 근사들)
    if k > 0:
        fig.add_scatter(x=xs[:k+1], y=np.zeros(k+1),
                        mode="markers+lines", name="x history",
                        line=dict(width=1, color="#666"))

    fig.update_layout(
        template="plotly_white", height=520,
        xaxis=dict(range=list(xr)),
        yaxis=dict(range=list(yr)),
        margin=dict(l=10, r=10, t=40, b=10),
        title=f"step {k} / {len(xs)-1}   ·   x_k ≈ {xn:.6f}"
    )
    return fig

def render():
    st.subheader("뉴턴법 (1D):  $x_{n+1}=x_n-\\frac{f(x_n)}{f'(x_n)}$  with  $f(x)=x^3-x-1$")
    # 파라미터
    x0    = st.slider("초기값 x₀", -2.0, 2.0, 0.5, 0.05, key=f"{PFX}:x0")
    steps = st.slider("최대 스텝 수", 2, 40, 12, key=f"{PFX}:steps")
    fps   = st.slider("FPS", 2, 30, 12, key=f"{PFX}:fps")
    xr    = st.slider("x-축 범위", -4.0, 4.0, (-2.2, 2.2), 0.1, key=f"{PFX}:xr")
    yr    = st.slider("y-축 범위", -6.0, 6.0, (-3.0, 3.0), 0.1, key=f"{PFX}:yr")

    # 이터레이션 미리 계산
    xs = _newton_iters(x0, steps)
    kmax = max(0, len(xs) - 1)

    # 미리보기 + 1회 재생 홀더
    holder = st.empty()

    # 미리보기 프레임 선택(정적)
    k_preview = st.slider("미리보기 step", 0, kmax, min(1, kmax), 1, key=f"{PFX}:kprev")
    xline = np.linspace(xr[0], xr[1], 1000)
    holder.plotly_chart(_draw_frame(xline, xs, k_preview, xr, yr), use_container_width=True)

    # 1회 재생 버튼
    if st.button("🎬 0 → 마지막 step까지 한 번 재생", key=f"{PFX}:once"):
        st.session_state[f"{PFX}:playing"] = True

    # 애니메이션 1회
    if st.session_state.get(f"{PFX}:playing", False):
        for k in range(0, kmax + 1):
            holder.plotly_chart(_draw_frame(xline, xs, k, xr, yr), use_container_width=True)
            time.sleep(1.0 / max(1, fps))
        st.session_state[f"{PFX}:playing"] = False

    # 수렴 요약
    st.markdown("---")
    st.caption(
        f"스텝 수: {len(xs)-1}  ·  마지막 근사값: **x ≈ {xs[-1]:.8f}**  ·  f(x) ≈ {f(xs[-1]):.2e}"
    )
