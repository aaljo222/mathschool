# tabs/tab_grad_descent.py
import numpy as np, streamlit as st, plotly.graph_objects as go
from utils.anim import next_frame_index

PFX = "gd2"

def render():
    st.subheader("경사하강법 (2D 쿼드라틱)")

    # f(x,y) = 1/2 (x,y)ᵀ A (x,y)
    a11 = st.slider("a11", 0.2, 4.0, 2.0, 0.1, key=f"{PFX}:a11")
    a22 = st.slider("a22", 0.2, 4.0, 0.8, 0.1, key=f"{PFX}:a22")
    a12 = st.slider("a12", -1.0, 1.0, 0.3, 0.05, key=f"{PFX}:a12")
    A = np.array([[a11, a12], [a12, a22]])

    x0 = np.array([st.slider("x₀", -3.0, 3.0, 2.0, 0.1, key=f"{PFX}:x0"),
                   st.slider("y₀", -3.0, 3.0, -1.5, 0.1, key=f"{PFX}:y0")])
    lr = st.slider("학습률 η", 0.01, 0.5, 0.12, 0.01, key=f"{PFX}:lr")

    fps   = st.slider("FPS", 2, 30, 12, key=f"{PFX}:fps")
    steps = st.slider("스텝 수", 5, 120, 60, key=f"{PFX}:steps")
    autorun = st.checkbox("자동 재생", True, key=f"{PFX}:auto")

    # 경로 미리 계산
    xs = [x0]
    for _ in range(steps-1):
        g = A @ xs[-1]
        xs.append(xs[-1] - lr * g)
    xs = np.array(xs)

    k = next_frame_index(PFX, steps, fps, autorun)

    X = np.linspace(-3,3,120)
    Y = np.linspace(-3,3,120)
    XX,YY = np.meshgrid(X,Y)
    ZZ = 0.5*(A[0,0]*XX**2 + 2*A[0,1]*XX*YY + A[1,1]*YY**2)

    fig = go.Figure(data=go.Contour(x=X, y=Y, z=ZZ, contours_coloring="lines", showscale=False))
    fig.add_scatter(x=xs[:k+1,0], y=xs[:k+1,1], mode="lines+markers", name="path", line=dict(width=3))
    fig.update_layout(template="plotly_white", height=520,
                      xaxis=dict(range=[-3,3]), yaxis=dict(range=[-3,3], scaleanchor="x", scaleratio=1),
                      title=f"step {k}/{steps-1}")
    st.plotly_chart(fig, use_container_width=True)
