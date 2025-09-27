# tabs/tab_grad_descent.py
import numpy as np, streamlit as st, plotly.graph_objects as go
from utils.anim import playbar, step_loop

def render():
    st.subheader("경사하강법(Gradient Descent) on f(x,y)=ax²+by²+cxy")
    a = st.slider("a", 0.1, 4.0, 1.5, 0.1)
    b = st.slider("b", 0.1, 4.0, 1.0, 0.1)
    c = st.slider("c (교차항)", -1.5, 1.5, 0.3, 0.1)
    lr = st.slider("학습률 η", 0.01, 0.5, 0.1, 0.01)
    iters = st.slider("스텝 수", 5, 200, 60, 5)
    start = np.array([st.slider("x₀", -3.0, 3.0, 2.5, 0.1),
                      st.slider("y₀", -3.0, 3.0, -2.0, 0.1)])

    def f(X):
        x, y = X[...,0], X[...,1]
        return a*x*x + b*y*y + c*x*y
    def g(xy):
        x, y = xy
        return np.array([2*a*x + c*y, 2*b*y + c*x])

    xs = np.linspace(-3,3,200); ys=np.linspace(-3,3,200)
    X,Y = np.meshgrid(xs,ys); Z = f(np.dstack([X,Y]))

    playing = playbar("gd")
    holder = st.empty()

    def draw(step):
        path = [start.copy()]
        for _ in range(step):
            path.append(path[-1] - lr*g(path[-1]))
        path = np.array(path)

        fig = go.Figure(data=go.Contour(x=xs, y=ys, z=Z, contours=dict(coloring="none")))
        fig.add_trace(go.Scatter(x=path[:,0], y=path[:,1], mode="lines+markers",
                                 name="경로", line=dict(color="#2c3e50")))
        fig.update_layout(template="plotly_white", height=520, xaxis_title="x", yaxis_title="y")
        holder.plotly_chart(fig, use_container_width=True)

    if playing:
        for i in step_loop(iters, fps=20, key="gd"):
            draw(i)
    else:
        draw(iters)
