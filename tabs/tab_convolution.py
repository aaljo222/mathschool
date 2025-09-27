import numpy as np, streamlit as st, plotly.graph_objects as go
from utils.anim import playbar, step_loop

PFX = "conv1d"

def render():
    st.subheader("1D 컨볼루션: 뒤집고 밀며 적분하기")

    L = 8.0
    t = np.linspace(-L, L, 1200)

    c1,c2,c3 = st.columns(3)
    with c1:
        a = st.slider("x(t) 중심", -3.0, 3.0, -1.0, 0.1, key=f"{PFX}:a")
        sx = st.slider("x 폭", 0.2, 2.5, 0.6, 0.05, key=f"{PFX}:sx")
    with c2:
        b = st.slider("h(t) 중심", -3.0, 3.0, 1.0, 0.1, key=f"{PFX}:b")
        sh = st.slider("h 폭", 0.2, 2.5, 0.8, 0.05, key=f"{PFX}:sh")
    with c3:
        fps  = st.slider("FPS", 2, 30, 12, key=f"{PFX}:fps")
        secs = st.slider("길이(초)", 1, 12, 6, key=f"{PFX}:secs")

    x = np.exp(-(t-a)**2/(2*sx**2))
    h = np.exp(-(t-b)**2/(2*sh**2))
    dt = t[1]-t[0]

    # 미리 전체 y(t) 계산
    y_full = np.convolve(x, h[::-1], mode="same")*dt

    holder = st.empty()
    playing = playbar(PFX)
    steps = max(2, int(secs*fps))

    def draw(frac):
        # τ = shift
        tau = (2*frac-1)*L
        h_flip_shift = np.interp(t, t[::-1]-tau, h, left=0, right=0)
        prod = x * h_flip_shift
        val  = prod.sum()*dt

        fig = go.Figure()
        fig.add_scatter(x=t, y=x, mode="lines", name="x(t)")
        fig.add_scatter(x=t, y=h_flip_shift, mode="lines", name="h(−τ)")
        fig.add_scatter(x=t, y=prod, mode="lines", name="곱", line=dict(width=1))
        fig.add_vline(x=tau, line_dash="dot")
        fig.update_layout(template="plotly_white", height=430,
                          title=f"τ={tau:.2f},  y(τ)≈{val:.3f}")
        holder.plotly_chart(fig, use_container_width=True)

        fig2 = go.Figure()
        fig2.add_scatter(x=t, y=y_full, mode="lines", name="y(t)=x∗h")
        fig2.update_layout(template="plotly_white", height=250)
        st.plotly_chart(fig2, use_container_width=True)

    if playing:
        for k in step_loop(steps, fps=fps, key=PFX):
            draw(k/(steps-1))
    else:
        draw(0.5)
