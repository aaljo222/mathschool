import numpy as np, streamlit as st, plotly.graph_objects as go
from utils.anim import playbar, step_loop

PFX = "gd2d"

def render():
    st.subheader("경사하강법 (2변수)")

    a = st.slider("a", 0.2, 3.0, 1.0, 0.1, key=f"{PFX}:a")
    b = st.slider("b", 0.2, 3.0, 2.0, 0.1, key=f"{PFX}:b")
    c = st.slider("교차항 c", -1.5, 1.5, 0.2, 0.05, key=f"{PFX}:c")
    eta = st.slider("학습률 η", 0.01, 1.0, 0.2, 0.01, key=f"{PFX}:eta")
    fps = st.slider("FPS", 2, 30, 12, key=f"{PFX}:fps")
    secs = st.slider("길이(초)", 1, 12, 6, key=f"{PFX}:secs")

    A = np.array([[a, c/2],[c/2, b]])
    # 양정치 보장
    A = A.T @ A + 0.1*np.eye(2)

    x0 = np.array([1.5, -1.0])
    holder = st.empty()
    playing = playbar(PFX)
    steps = max(2, int(secs*fps))

    def f(x): return 0.5 * x @ A @ x
    def g(x): return A @ x

    # 등고선용 격자
    xs = np.linspace(-2.5, 2.5, 120)
    ys = np.linspace(-2.5, 2.5, 120)
    X,Y = np.meshgrid(xs, ys)
    Z = 0.5*(A[0,0]*X**2 + 2*A[0,1]*X*Y + A[1,1]*Y**2)

    def draw(t):
        k = int(t*(steps-1))
        path = [x0.copy()]
        x = x0.copy()
        for _ in range(k):
            x -= eta * g(x)
            path.append(x.copy())
        P = np.array(path)

        fig = go.Figure(data=go.Contour(x=xs, y=ys, z=Z, contours=dict(showlabels=True)))
        fig.add_scatter(x=P[:,0], y=P[:,1], mode="lines+markers", name="path",
                        line=dict(width=3))
        fig.update_layout(template="plotly_white", height=520,
                          xaxis_title="x", yaxis_title="y")
        holder.plotly_chart(fig, use_container_width=True)

    if playing:
        for k in step_loop(steps, fps=fps, key=PFX):
            draw(k/(steps-1))
    else:
        draw(0.5)
