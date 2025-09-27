import numpy as np, streamlit as st, plotly.graph_objects as go
from utils.anim import playbar, step_loop

PFX = "fs_series"

def _target(x, kind):
    if kind == "사각파":
        return np.sign(np.sin(x))
    if kind == "톱니파":
        return (x/np.pi % 2) - 1
    return 2/np.pi*np.arcsin(np.sin(x))  # 삼각파

def render():
    st.subheader("푸리에 급수 부분합으로 근사하기")

    kind = st.selectbox("대상", ["사각파", "톱니파", "삼각파"], key=f"{PFX}:kind")
    Nmax = st.slider("최대 항 수 N", 1, 60, 30, key=f"{PFX}:Nmax")
    fps  = st.slider("FPS", 2, 30, 12, key=f"{PFX}:fps")
    secs = st.slider("길이(초)", 1, 12, 6, key=f"{PFX}:secs")

    x = np.linspace(-np.pi, np.pi, 1500)
    f = _target(x, kind)

    playing = playbar(PFX)
    holder  = st.empty()
    steps = max(2, int(secs*fps))

    def partial(N):
        # 표준 형태의 급수(기본주파수 1)
        y = np.zeros_like(x)
        if kind == "사각파":
            ks = np.arange(1, N+1, 2)
            for k in ks: y += (4/np.pi)*(1/k)*np.sin(k*x)
        elif kind == "톱니파":
            ks = np.arange(1, N+1)
            for k in ks: y += (-2/np.pi)*(1/k)*np.sin(k*x)
        else:  # 삼각파
            ks = np.arange(1, N+1, 2)
            for k in ks: y += (8/np.pi**2)*(1/k**2)*((-1)**((k-1)//2))*np.sin(k*x)
        return y

    def draw(t):
        N = max(1, int(1 + t*(Nmax-1)))
        y = partial(N)
        fig = go.Figure()
        fig.add_scatter(x=x, y=f, mode="lines", name="target")
        fig.add_scatter(x=x, y=y, mode="lines", name=f"sum_1..{N}", line=dict(width=3))
        fig.update_layout(template="plotly_white", height=470,
                          title=f"{kind}  —  부분합 N={N}")
        holder.plotly_chart(fig, use_container_width=True)

    if playing:
        for k in step_loop(steps, fps=fps, key=PFX):
            draw(k/(steps-1))
    else:
        draw(0.2)
