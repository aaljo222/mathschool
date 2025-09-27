# tabs/tab_newton.py
import numpy as np, streamlit as st, plotly.graph_objects as go
from utils.anim import next_frame_index

PFX = "newton1d"

def render():
    st.subheader("뉴턴법 (1D)")

    st.latex(r"f(x)=x^3 - x - 1,\quad x_{n+1}=x_n - \frac{f(x_n)}{f'(x_n)}")
    x0 = st.slider("초기값 x₀", -2.0, 2.0, 0.5, 0.05, key=f"{PFX}:x0")

    fps   = st.slider("FPS", 2, 30, 12, key=f"{PFX}:fps")
    steps = st.slider("스텝 수", 2, 40, 12, key=f"{PFX}:steps")
    autorun = st.checkbox("자동 재생", True, key=f"{PFX}:auto")

    f  = lambda x: x**3 - x - 1
    fp = lambda x: 3*x**2 - 1

    xs = [x0]
    for _ in range(steps-1):
        xn = xs[-1]
        xs.append(xn - f(xn)/ (fp(xn) + 1e-12))
    xs = np.array(xs)

    k = next_frame_index(PFX, steps, fps, autorun)
    xline = np.linspace(-2.2, 2.2, 800)

    fig = go.Figure()
    fig.add_scatter(x=xline, y=f(xline), mode="lines", name="f(x)")
    fig.add_hline(y=0, line_color="#888")

    # 현재 점과 다음 접선
    xn = xs[k]
    fig.add_scatter(x=[xn], y=[f(xn)], mode="markers", name=f"x{k}")
    # 접선 y = f(xn) + f'(xn)(x-xn)
    ytan = f(xn) + fp(xn)*(xline - xn)
    fig.add_scatter(x=xline, y=ytan, mode="lines", name="tangent", line=dict(dash="dot"))

    fig.update_layout(template="plotly_white", height=520,
                      xaxis=dict(range=[-2.2,2.2]),
                      yaxis=dict(range=[-3,3]),
                      title=f"step {k}/{steps-1},  x_k ≈ {xn:.6f}")
    st.plotly_chart(fig, use_container_width=True)
