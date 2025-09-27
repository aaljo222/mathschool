# tabs/tab_newton.py
import numpy as np, streamlit as st, plotly.graph_objects as go
from utils.anim import playbar, step_loop

def _func(name):
    if name == "x^3-2x-2":
        f  = lambda x: x**3 - 2*x - 2
        fp = lambda x: 3*x**2 - 2
    elif name == "cos x - x":
        f  = lambda x: np.cos(x) - x
        fp = lambda x: -np.sin(x) - 1
    else:  # x^2-2
        f  = lambda x: x**2 - 2
        fp = lambda x: 2*x
    return f, fp

def render():
    st.subheader("뉴턴법: 접선으로 뿌리 찾기 (코블웹)")
    fn = st.selectbox("함수 선택", ["x^2-2", "x^3-2x-2", "cos x - x"])
    f, fp = _func(fn)
    x0 = st.slider("초기값 x₀", -3.0, 3.0, 1.0, 0.1)
    iters = st.slider("반복 횟수", 1, 20, 8)
    rng = st.slider("표시 구간", -3.0, 3.0, (-2.5, 2.5))

    xs = np.linspace(*rng, 600)
    figph = st.empty()
    playing = playbar("newton")

    def draw(step):
        x = x0
        points = [x]
        for _ in range(step):
            x = x - f(x)/fp(x)
            points.append(x)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xs, y=f(xs), mode="lines", name="f(x)"))
        fig.add_hline(y=0, line_color="#aaa")
        fig.update_layout(template="plotly_white", height=480, xaxis_title="x", yaxis_title="y")
        # 접선/수직 보조선
        for k in range(min(step, len(points)-1)):
            xk = points[k]; yk = f(xk)
            slope = fp(xk)
            xt = np.linspace(xk-1, xk+1, 20)
            fig.add_trace(go.Scatter(x=[xk, xk], y=[0, yk], mode="lines", line=dict(dash="dot"), showlegend=False))
            fig.add_trace(go.Scatter(x=xt, y=yk + slope*(xt-xk), mode="lines", line=dict(color="#d35400"), showlegend=False))
            fig.add_trace(go.Scatter(x=[points[k+1], points[k+1]], y=[0, f(points[k+1])], mode="lines",
                                     line=dict(dash="dot"), showlegend=False))
            fig.add_trace(go.Scatter(x=[xk], y=[yk], mode="markers", marker=dict(size=8), showlegend=False))
        figph.plotly_chart(fig, use_container_width=True)

    if playing:
        for i in step_loop(iters+1, fps=2, key="newton"):
            draw(i)
    else:
        draw(iters)
