import numpy as np, streamlit as st, plotly.graph_objects as go
from utils.anim import playbar, step_loop

PFX = "newton1d"

def render():
    st.subheader("뉴턴 방법(1변수) — 접선으로 뿌리 찾기")

    func = st.selectbox("함수", ["x^3 - x - 1", "cos x - x/2", "x - tanh(2x)"],
                        key=f"{PFX}:f")
    x0   = st.slider("초기값 x₀", -3.0, 3.0, 1.0, 0.05, key=f"{PFX}:x0")
    fps  = st.slider("FPS", 2, 30, 12, key=f"{PFX}:fps")
    secs = st.slider("길이(초)", 1, 12, 6, key=f"{PFX}:secs")
    steps = max(2, int(secs*fps))

    if func == "x^3 - x - 1":
        f  = lambda x: x**3 - x - 1
        df = lambda x: 3*x**2 - 1
        xr = 1.32
    elif func == "cos x - x/2":
        f  = lambda x: np.cos(x) - x/2
        df = lambda x: -np.sin(x) - 0.5
        xr = 0.0
    else:
        f  = lambda x: x - np.tanh(2*x)
        df = lambda x: 1 - 2*(1/np.cosh(2*x))**2
        xr = 0.0

    xs = np.linspace(-3, 3, 1000)
    holder = st.empty()
    playing = playbar(PFX)

    def iterate(kmax):
        pts = [x0]
        x = x0
        for _ in range(kmax):
            x = x - f(x)/df(x)
            pts.append(x)
        return np.array(pts)

    def draw(t):
        k = int(t*(steps-1))
        P = iterate(k)

        fig = go.Figure()
        fig.add_scatter(x=xs, y=f(xs), mode="lines", name="f(x)")
        # 접선들
        for i in range(len(P)-1):
            xk = P[i]
            yk = f(xk)
            slope = df(xk)
            yline = yk + slope*(xs - xk)
            fig.add_scatter(x=xs, y=yline, mode="lines",
                            line=dict(dash="dot"), name=f"tangent {i+1}", showlegend=False)
            fig.add_scatter(x=[xk], y=[yk], mode="markers", showlegend=False)

        fig.add_hline(y=0, line_color="#999")
        fig.update_layout(template="plotly_white", height=520,
                          title=f"k={k},  근사해 ≈ {P[-1]:.6f} (참고: {xr:.3f})",
                          xaxis_title="x", yaxis_title="f(x)")
        holder.plotly_chart(fig, use_container_width=True)

    if playing:
        for i in step_loop(steps, fps=fps, key=PFX):
            draw(i/(steps-1))
    else:
        draw(0.3)
