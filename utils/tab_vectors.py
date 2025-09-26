# tabs/tab_vectors.py
import streamlit as st
import numpy as np
import plotly.graph_objects as go

def render():
    st.subheader("벡터의 선형결합:  a·v₁ + b·v₂")

    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        st.markdown("**v₁**")
        v1x = st.number_input("v₁ₓ", value=2.0, step=0.1, key="v1x")
        v1y = st.number_input("v₁ᵧ", value=1.0, step=0.1, key="v1y")
    with c2:
        st.markdown("**v₂**")
        v2x = st.number_input("v₂ₓ", value=1.0, step=0.1, key="v2x")
        v2y = st.number_input("v₂ᵧ", value=2.0, step=0.1, key="v2y")
    with c3:
        mode = st.radio("모드", ["애니메이션 (a=cos t, b=sin t)", "수동 a,b"], index=0)
        show_grid = st.checkbox("격자 표시", value=True)
        keep_ratio = st.checkbox("축 비율 1:1", value=True)

    v1 = np.array([v1x, v1y], dtype=float)
    v2 = np.array([v2x, v2y], dtype=float)

    max_len = max(1.0, float(np.linalg.norm(v1) + np.linalg.norm(v2)))
    rng = float(np.ceil(max_len + 0.5))
    xr, yr = [-rng, rng], [-rng, rng]

    fig = go.Figure()

    if mode.startswith("수동"):
        a = st.slider("a", -3.0, 3.0, 1.0, 0.1)
        b = st.slider("b", -3.0, 3.0, 1.0, 0.1)
        r = (a*v1 + b*v2).astype(float)

        fig.add_trace(go.Scatter(x=[0, v1[0]], y=[0, v1[1]], mode="lines+markers", name="v1"))
        fig.add_trace(go.Scatter(x=[0, v2[0]], y=[0, v2[1]], mode="lines+markers", name="v2"))
        fig.add_trace(go.Scatter(x=[0, r[0]],  y=[0, r[1]],  mode="lines+markers", name="a·v1 + b·v2"))
        fig.add_trace(go.Scatter(x=[r[0]], y=[r[1]], mode="markers", name="locus", showlegend=False))
        fig.add_trace(go.Scatter(
            x=[0, v1[0], r[0], v2[0], 0], y=[0, v1[1], r[1], v2[1], 0],
            fill="toself", mode="lines", name="parallelogram", showlegend=False, opacity=0.2
        ))
        fig.update_layout(title=f"a={a:.2f}, b={b:.2f}")

    else:
        T = st.slider("프레임 수", 30, 240, 120, 10)
        speed = st.slider("속도 (ms/프레임)", 10, 200, 40, 5)
        t = np.linspace(0, 2*np.pi, int(T))
        a = np.cos(t); b = np.sin(t)
        res = (np.outer(a, v1) + np.outer(b, v2)).astype(float)
        r0 = res[0]
        fig.add_trace(go.Scatter(x=[0, v1[0]], y=[0, v1[1]], mode="lines+markers", name="v1"))
        fig.add_trace(go.Scatter(x=[0, v2[0]], y=[0, v2[1]], mode="lines+markers", name="v2"))
        fig.add_trace(go.Scatter(x=[0, r0[0]], y=[0, r0[1]], mode="lines+markers", name="a·v1 + b·v2"))
        fig.add_trace(go.Scatter(x=[r0[0]], y=[r0[1]], mode="markers", name="locus", showlegend=False))

        frames = []
        for i in range(int(T)):
            rx, ry = float(res[i,0]), float(res[i,1])
            frames.append(go.Frame(
                data=[go.Scatter(), go.Scatter(),
                      go.Scatter(x=[0, rx], y=[0, ry]),
                      go.Scatter(x=res[:i+1,0].tolist(), y=res[:i+1,1].tolist())],
                name=f"t{i}",
                layout=go.Layout(title=f"a=cos t={a[i]:.2f},  b=sin t={b[i]:.2f}")
            ))
        fig.frames = frames
        fig.update_layout(
            updatemenus=[{
                "type":"buttons",
                "buttons":[
                    {"label":"▶ Play","method":"animate",
                     "args":[None,{"frame":{"duration":int(speed),"redraw":True},
                                   "fromcurrent":True,"transition":{"duration":0}}]},
                    {"label":"⏸ Pause","method":"animate",
                     "args":[[None],{"mode":"immediate","frame":{"duration":0,"redraw":False},
                                     "transition":{"duration":0}}]}
                ],
                "direction":"left","x":0.0,"y":1.12
            }],
            sliders=[{
                "steps":[{"args":[[f"t{i}"],{"frame":{"duration":0,"redraw":True},
                                             "mode":"immediate","transition":{"duration":0}}],
                          "label":f"{i}","method":"animate"} for i in range(int(T))],
                "x":0.05,"y":1.04,"len":0.9
            }]
        )

    fig.update_layout(
        xaxis=dict(range=xr, showgrid=show_grid, zeroline=True),
        yaxis=dict(range=yr, showgrid=show_grid, zeroline=True),
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(bgcolor="rgba(255,255,255,0.6)")
    )
    if keep_ratio:
        fig.update_yaxes(scaleanchor="x", scaleratio=1)

    st.plotly_chart(fig, use_container_width=True)
