# tabs/tab_lincomb3.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# LaTeX 헬퍼(모듈화)
from utils import latex as L


# ────────────────────────────── 유틸 ──────────────────────────────
@dataclass
class V3:
    x: float; y: float; z: float
    def as_np(self): return np.array([self.x, self.y, self.z], dtype=float)

def _arrow3d(p0, p1, name, color, width=6, showlegend=True):
    x0,y0,z0 = p0; x1,y1,z1 = p1
    return [
        go.Scatter3d(x=[x0, x1], y=[y0, y1], z=[z0, z1],
                     mode="lines", line=dict(width=width, color=color),
                     name=name, showlegend=showlegend),
        go.Scatter3d(x=[x1], y=[y1], z=[z1],
                     mode="markers", marker=dict(size=4, color=color),
                     name=f"{name}·tip", showlegend=False),
    ]

def _axes(length=1.0):
    Lp = length
    return [
        go.Scatter3d(x=[0,Lp], y=[0,0], z=[0,0], mode="lines",
                     line=dict(color="#e74c3c", width=2), name="x"),
        go.Scatter3d(x=[0,0], y=[0,Lp], z=[0,0], mode="lines",
                     line=dict(color="#27ae60", width=2), name="y"),
        go.Scatter3d(x=[0,0], y=[0,0], z=[0,Lp], mode="lines",
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

def _path_until(vs, xs, order, steps_each, progress):
    """
    order에 정한 순서대로 x_i v_i를 누적.
    steps_each * len(order) 개의 가상 스텝 중 progress(0..1)에 해당하는 지점까지만 경로/현재점 계산
    """
    v1,v2,v3 = vs
    x1,x2,x3 = xs
    pool = {"v1": (v1,x1), "v2": (v2,x2), "v3": (v3,x3)}
    segs = [pool[tag] for tag in order if tag in pool]

    total_steps = max(1, steps_each * len(segs))
    kmax = int(np.clip(progress, 0.0, 1.0) * total_steps)

    origin = np.zeros(3)
    cur = origin.copy()
    hist = [cur.copy()]
    done_steps = 0

    for v, coef in segs:
        for k in range(1, steps_each+1):
            if done_steps >= kmax:
                # 여기서 멈추면 이전 cur에서 현재 세그먼트의 비율만큼만 진행
                t_partial = 0.0 if steps_each == 0 else (kmax - (done_steps)) / steps_each
                t_partial = np.clip(t_partial, 0.0, 1.0)
                cur = cur + (t_partial * coef) * v
                hist.append(cur.copy())
                return hist, cur
            # 계속 진행
            t = k / steps_each
            nxt = cur + (t * coef) * v
            hist.append(nxt.copy())
            done_steps += 1
        cur = cur + coef * v

    # progress=1.0이면 완주
    return hist, cur


# ────────────────────────────── 메인 ──────────────────────────────
def render():
    st.subheader("3×3 연립방정식  /  열벡터 선형결합 (이벤트 1회 갱신)")

    # 좌측 설정
    c0,c1,c2 = st.columns([1.2,1.2,1.0])
    with c0:
        preset = st.selectbox("프리셋",
            ["Orthonormal(I)", "Skewed", "Nearly singular", "Custom"])
    with c1:
        steps_each = st.slider("세그먼트 단계(궤적 해상도)", 5, 100, 30, 5)
    with c2:
        progress = st.slider("진행도", 0.0, 1.0, 1.0, 0.01,
                             help="선형결합 누적합을 이 비율만큼 진행한 지점까지 그립니다.")

    # 열벡터 v1,v2,v3 (프리셋/커스텀)
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

    # 해 x (가역/비가역)
    detA = float(np.linalg.det(A))
    if abs(detA) > 1e-10:
        x = np.linalg.solve(A, b); res = None
        solv_text = "가역 행렬 → 유일해"
    else:
        x, *_ = np.linalg.lstsq(A, b, rcond=None)
        res = float(np.linalg.norm(A @ x - b))
        solv_text = "비가역/근사해(최소제곱)"

    # ── 표기 모드 선택 + LaTeX 출력 ──
    show_mode = st.radio("표시 형식", ["행렬식 A·x=b", "연립방정식", "선형결합식"], horizontal=True)
    if show_mode == "행렬식 A·x=b":
        st.latex(L.axb(A, b, with_labels=True, det=detA, residual=res, digits=3))
        st.latex(L.vector("x", x, digits=3))
    elif show_mode == "연립방정식":
        st.latex(L.system(A, b, digits=3))
        st.latex(L.vector("x", x, digits=3))
    else:
        st.latex(L.lincomb("Ax", x, basis=(r"v_1", r"v_2", r"v_3"), digits=3))

    st.caption(solv_text + "   ·   x = " + ", ".join(f"{t:.3g}" for t in x))

    # ── 더할 벡터 선택/순서 ──
    order = st.multiselect("더할 벡터와 순서(위에서 아래 순서대로)", ["v1","v2","v3"],
                           default=["v1","v2","v3"])
    if not order:
        st.info("하나 이상의 벡터를 선택하세요. (예: v1, v2, v3)")
        order = ["v1","v2","v3"]

    # ── 정지 요소 + 부분합 경로(단일 렌더) ──
    all_pts = np.column_stack([np.zeros(3), *vs, b, A@x])
    R = float(np.max(np.abs(all_pts))) * 1.3 or 1.0
    rng = [-R, R]

    fig = go.Figure()
    for tr in _axes(R): fig.add_trace(tr)
    fig.add_traces(_arrow3d([0,0,0], vs[0], "v1", "#8e44ad"))
    fig.add_traces(_arrow3d([0,0,0], vs[1], "v2", "#16a085"))
    fig.add_traces(_arrow3d([0,0,0], vs[2], "v3", "#d35400"))
    fig.add_traces(_arrow3d([0,0,0], b,   "b (target)", "#c0392b"))
    fig.add_traces(_arrow3d([0,0,0], A@x, "Ax", "#2c3e50"))

    # 진행도에 맞는 경로 계산(프레임 없음)
    hist, cur = _path_until(vs, x, order=tuple(order), steps_each=steps_each, progress=progress)
    hx, hy, hz = (np.array(hist)[:,0], np.array(hist)[:,1], np.array(hist)[:,2])

    # 회색 궤적 + 현재 Σ
    fig.add_trace(go.Scatter3d(x=hx, y=hy, z=hz, mode="lines",
                               line=dict(width=4, color="#7f8c8d"),
                               name="partial sum"))
    fig.add_traces(_arrow3d([0,0,0], cur, "Σ", "#2c3e50", width=8, showlegend=True))

    fig.update_layout(
        **_base_layout(rng),
        title="열벡터 선형결합:  Σ = x₁v₁ + x₂v₂ + x₃v₃  (회색 궤적 · 단일 렌더)"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        """
**설명**  
- \(A=[v_1\ v_2\ v_3]\), \(Ax = x_1v_1 + x_2v_2 + x_3v_3\).  
- ‘진행도’ 슬라이더로 \(x_i v_i\)의 누적합을 원하는 비율까지만 그립니다(자동재생/버튼 없음).  
- \(\det(A)\neq 0\) 이면 유일해, 그렇지 않으면 최소제곱 해를 표시합니다.
        """
    )
