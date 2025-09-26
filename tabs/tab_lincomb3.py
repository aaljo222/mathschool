# tabs/tab_lincomb3.py
from __future__ import annotations
import itertools
from dataclasses import dataclass
import numpy as np
import streamlit as st
import plotly.graph_objects as go

@dataclass
class V3:
    x: float; y: float; z: float
    def as_np(self): return np.array([self.x, self.y, self.z], dtype=float)

def _arrow3d(p0, p1, name, color, width=6, showlegend=True):
    """단순 3D 화살표(선+마커)"""
    x0,y0,z0 = p0; x1,y1,z1 = p1
    return [
        go.Scatter3d(x=[x0, x1], y=[y0, y1], z=[z0, z1],
                     mode="lines", line=dict(width=width, color=color),
                     name=name, showlegend=showlegend),
        go.Scatter3d(x=[x1], y=[y1], z=[z1], mode="markers",
                     marker=dict(size=4, color=color), name=f"{name}·tip",
                     showlegend=False),
    ]

def _axes(length=1.0):
    L=length
    return [
        go.Scatter3d(x=[0,L], y=[0,0], z=[0,0], mode="lines",
                     line=dict(color="#e74c3c", width=2), name="x"),
        go.Scatter3d(x=[0,0], y=[0,L], z=[0,0], mode="lines",
                     line=dict(color="#27ae60", width=2), name="y"),
        go.Scatter3d(x=[0,0], y=[0,0], z=[0,L], mode="lines",
                     line=dict(color="#2980b9", width=2), name="z"),
    ]

def _base_layout(rng):
    return dict(
        scene=dict(
            xaxis=dict(range=rng, showspikes=False, zeroline=True),
            yaxis=dict(range=rng, showspikes=False, zeroline=True),
            zaxis=dict(range=rng, showspikes=False, zeroline=True),
            aspectmode="cube",
        ),
        template="plotly_white",
        margin=dict(l=0, r=0, t=40, b=0),
        height=560,
        legend=dict(bgcolor="rgba(255,255,255,.6)"),
    )

def _latex_Axb(A,b,x,detA,res=None):
    def row(v): return " & ".join(f"{t:.2g}" for t in v)
    Atex = r"\begin{bmatrix}" + r"\\ ".join(row(r) for r in A) + r"\end{bmatrix}"
    xtex = r"\begin{bmatrix}" + r"\\ ".join(f"{t:.2g}" for t in x) + r"\end{bmatrix}"
    btex = r"\begin{bmatrix}" + r"\\ ".join(f"{t:.2g}" for t in b) + r"\end{bmatrix}"
    det = f"{detA:.3g}"
    s  = rf" A\,x=b\quad:\quad {Atex}\,{xtex} \;=\; {btex}\qquad \det(A)={det}"
    if res is not None:
        s += rf"\quad,\; \|Ax-b\|_2={res:.2e}"
    return "$" + s + "$"

def _make_frames(vs, xs, order=("v1","v2","v3"), steps_each=30):
    """순차 합 애니메이션 프레임 생성"""
    v1,v2,v3 = vs
    x1,x2,x3 = xs
    segs = [("v1", v1, x1), ("v2", v2, x2), ("v3", v3, x3)]
    segs = [s for s in segs if s[0] in order]

    frames=[]
    origin = np.zeros(3)
    cur = origin.copy()
    hist = [cur.copy()]

    for tag, v, coef in segs:
        for k in range(1, steps_each+1):
            t = k/steps_each
            nxt = cur + (t*coef)*v
            hist.append(nxt.copy())
            frames.append(go.Frame(
                name=f"{tag}-{k}",
                data=[
                    # locus so far
                    go.Scatter3d(x=[p[0] for p in hist], y=[p[1] for p in hist], z=[p[2] for p in hist],
                                 mode="lines", line=dict(width=4, color="#7f8c8d"), name="partial sum",
                                 showlegend=False),
                    # running sum arrow
                    *_arrow3d(origin, nxt, "Σ", "#2c3e50", width=8, showlegend=False),
                ]
            ))
        cur = cur + coef*v
    return frames, hist[-1]

def render():
    st.subheader("3×3 연립방정식  /  열벡터 선형결합 애니메이션")

    # ── 좌측: 입력 ────────────────────────────────────────────────────────
    c0,c1,c2 = st.columns([1.2,1.2,1.0])

    with c0:
        preset = st.selectbox("프리셋",
            ["Orthonormal(I)", "Skewed", "Nearly singular", "Custom"])
    with c1:
        secs = st.slider("길이(초)", 2, 12, 6, 1)
        fps  = st.slider("FPS", 10, 40, 24, 2)
    with c2:
        steps_each = st.slider("세그먼트 단계", 10, 80, 30, 5,
                               help="v1→v2→v3로 더할 때 각 구간의 프레임 수")

    # 프리셋 or 커스텀 A (열벡터 v1,v2,v3)
    if preset == "Orthonormal(I)":
        v1 = V3(1,0,0); v2=V3(0,1,0); v3=V3(0,0,1)
    elif preset == "Skewed":
        v1 = V3(1,0.2,0.1); v2=V3(0.3,1,0.2); v3=V3(0.1,0.3,1)
    elif preset == "Nearly singular":
        v1 = V3(1,0,0); v2=V3(1.01,0.01,0); v3=V3(0,0,1e-2)
    else:
        cc1,cc2,cc3 = st.columns(3)
        with cc1:
            v1 = V3(st.number_input("v1₁", value=1.0), st.number_input("v1₂", value=0.0), st.number_input("v1₃", value=0.0))
        with cc2:
            v2 = V3(st.number_input("v2₁", value=0.0), st.number_input("v2₂", value=1.0), st.number_input("v2₃", value=0.0))
        with cc3:
            v3 = V3(st.number_input("v3₁", value=0.0), st.number_input("v3₂", value=0.0), st.number_input("v3₃", value=1.0))

    vs = [v1.as_np(), v2.as_np(), v3.as_np()]
    A  = np.column_stack(vs)

    # b 입력
    d1,d2,d3 = st.columns(3)
    with d1: b1 = st.number_input("b₁", value=1.0, step=0.1)
    with d2: b2 = st.number_input("b₂", value=0.8, step=0.1)
    with d3: b3 = st.number_input("b₃", value=0.6, step=0.1)
    b = np.array([b1,b2,b3], float)

    # 해 x 구하기 (가역/비가역 대응)
    detA = float(np.linalg.det(A))
    if abs(detA) > 1e-10:
        x = np.linalg.solve(A, b)
        res = None
        solv_text = "가역 행렬 → 유일해"
    else:
        x, *_ = np.linalg.lstsq(A, b, rcond=None)
        res = float(np.linalg.norm(A@x - b))
        solv_text = "비가역/근사해(최소제곱)"

    st.latex(_latex_Axb(A, b, x, detA, res))
    st.caption(solv_text + "   ·   x = " + ", ".join(f"{t:.3g}" for t in x))

    # ── 3D 그림(정지요소) ────────────────────────────────────────────────
    # 축 범위
    all_pts = np.column_stack([np.zeros(3), *vs, b, A@x])
    R = float(np.max(np.abs(all_pts))) * 1.3 or 1.0
    rng = [-R, R]

    fig = go.Figure()
    for tr in _axes(R): fig.add_trace(tr)

    # 열벡터 v1,v2,v3
    fig.add_traces(_arrow3d([0,0,0], vs[0], "v1", "#8e44ad"))
    fig.add_traces(_arrow3d([0,0,0], vs[1], "v2", "#16a085"))
    fig.add_traces(_arrow3d([0,0,0], vs[2], "v3", "#d35400"))

    # 목표 b와 해 Ax
    fig.add_traces(_arrow3d([0,0,0], b,    "b (target)", "#c0392b"))
    fig.add_traces(_arrow3d([0,0,0], A@x,  "Ax", "#2c3e50"))

    # ── 프레임(Σ 부분합 경로) ─────────────────────────────────────────────
    frames, last = _make_frames(vs, x, steps_each=steps_each)
    fig.frames = frames

    # updatemenus / slider
    total = len(frames)
    fig.update_layout(
        **_base_layout(rng),
        title="열벡터 선형결합:  Σ = x₁v₁ + x₂v₂ + x₃v₃  (회색 궤적)",
        updatemenus=[{
            "type":"buttons", "direction":"left", "x":0.0, "y":1.08,
            "buttons":[
                {"label":"▶ Play","method":"animate",
                 "args":[None,{"frame":{"duration": int(1000*secs/max(total,1)/fps), "redraw":True},
                               "fromcurrent":True,"transition":{"duration":0}}]},
                {"label":"⏸ Pause","method":"animate",
                 "args":[[None],{"mode":"immediate",
                                 "frame":{"duration":0,"redraw":False},
                                 "transition":{"duration":0}}]}
            ]
        }],
        sliders=[{
            "x":0.05,"y":1.02,"len":0.9,
            "steps":[{"args":[[f.name],
                              {"mode":"immediate",
                               "frame":{"duration":0,"redraw":True},
                               "transition":{"duration":0}}],
                      "label":str(i+1),"method":"animate"} for i,f in enumerate(frames)]
        }]
    )

    st.plotly_chart(fig, use_container_width=True)

    # 부가 정보
    st.markdown(
        """
**설명**  
- \(A=[v_1\ v_2\ v_3]\), \(Ax = x_1v_1 + x_2v_2 + x_3v_3\).  
- 회색 궤적은 \(x_1v_1\) → \(+x_2v_2\) → \(+x_3v_3\) 순서로 누적합을 그립니다.  
- \(\det(A)\neq 0\) 이면 유일해, 그렇지 않으면 최소제곱 해를 표시합니다.
        """
    )
