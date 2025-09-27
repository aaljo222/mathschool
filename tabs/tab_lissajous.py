# tabs/tab_lissajous.py
import math
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from utils.anim import next_frame_index

PFX = "liss"

def render():
    st.subheader("리사주 곡선")

    col = st.columns(5)
    with col[0]: a = st.slider("a", 1, 12, 3, key=f"{PFX}:a")
    with col[1]: b = st.slider("b", 1, 12, 2, key=f"{PFX}:b")
    with col[2]: d = st.slider("δ (rad)", 0.0, 2*np.pi, np.pi/2, 0.01, key=f"{PFX}:d")
    with col[3]: fps   = st.slider("FPS", 2, 30, 12, key=f"{PFX}:fps")
    with col[4]: steps = st.slider("프레임 수", 20, 240, 120, key=f"{PFX}:steps")

    # 자동 재생(체크=움직임 / 해제=정지 프레임)
    autorun = st.checkbox("자동 재생", True, key=f"{PFX}:auto")

    # 진행도(0~1)에 대응하는 프레임 인덱스
    k = next_frame_index(PFX, steps, fps, autorun)
    t_frac = k / max(1, steps - 1)          # 0~1

    # 파라미터 & 좌표
    tt = np.linspace(0, 2*np.pi, 2000)      # 전체 곡선
    x = np.sin(a*tt + d)
    y = np.sin(b*tt)

    # 진행도만큼 그리기(부분 궤적)
    t_end = t_frac * 2*np.pi
    mask = tt <= t_end
    px, py = np.sin(a*t_end + d), np.sin(b*t_end)

    # 기약비/폐곡선 힌트
    g = math.gcd(a, b)
    a_s, b_s = a // g, b // g
    ratio_text = f"a:b = {a}:{b} (기약 {a_s}:{b_s}) — 정수비면 폐곡선을 이룹니다."

    fig = go.Figure()
    # 전체 윤곽(연한 회색)
    fig.add_scatter(x=x, y=y, mode="lines", line=dict(width=1, color="rgba(0,0,0,0.18)"),
                    name="전체 윤곽", showlegend=False)
    # 진행 궤적(진한 색)
    fig.add_scatter(x=x[mask], y=y[mask], mode="lines", line=dict(width=3),
                    name="진행 궤적")
    # 현재점
    fig.add_scatter(x=[px], y=[py], mode="markers",
                    marker=dict(size=10), name="현재 위치")

    fig.update_layout(
        template="plotly_white",
        height=520,
        title=f"리사주: x=sin(a·t+δ), y=sin(b·t)   |   t={t_frac:.2f},  δ={d:.2f} rad",
        xaxis=dict(range=[-1.1, 1.1], zeroline=True),
        yaxis=dict(range=[-1.1, 1.1], zeroline=True, scaleanchor="x", scaleratio=1),
        legend=dict(bgcolor="rgba(255,255,255,0.65)")
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(ratio_text)
