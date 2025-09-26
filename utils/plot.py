# utils/plot.py
from __future__ import annotations

import math
import numpy as np
import plotly.graph_objects as go

__all__ = [
    "line_fig",
    "contour_implicit",
    "phasor_fig",
    "waveform_fig",
    "make_parallel_animation",
]

# ---------------------------------------------------------------------
# 공용 2D 라인/등고선 유틸
# ---------------------------------------------------------------------
def line_fig(x, ys, names, title, xaxis="x", yaxis="y"):
    """여러 개의 y(x) 곡선을 한 Figure에 그리기"""
    fig = go.Figure()
    for y, name in zip(ys, names):
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name))
    fig.update_layout(
        title=title, xaxis_title=xaxis, yaxis_title=yaxis, template="plotly_white"
    )
    return fig


def contour_implicit(F, x_range, y_range, level=0.0, title=""):
    """암시적 곡선 F(x,y)=level 을 등고선으로 그리기"""
    xs = np.linspace(*x_range, 600)
    ys = np.linspace(*y_range, 600)
    X, Y = np.meshgrid(xs, ys)
    Z = F(X, Y)
    fig = go.Figure(
        data=go.Contour(
            x=xs,
            y=ys,
            z=Z,
            contours=dict(start=level, end=level, size=1, coloring="none"),
            line_width=2,
            showscale=False,
        )
    )
    fig.update_layout(title=title, xaxis_title="x", yaxis_title="y", template="plotly_white")
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


# ---------------------------------------------------------------------
# 전기 파형/위상 유틸
# ---------------------------------------------------------------------
def phasor_fig(V: float, I: float, phi: float, title: str = "Phasor"):
    """
    위상자(복소평면) 그림.
    V, I는 스케일 참고값(단위 원으로 그리므로 길이엔 직접 사용하지 않음),
    phi는 전압→전류 위상차(rad).
    """
    th = np.linspace(0, 2 * np.pi, 360)
    fig = go.Figure()
    # 단위원
    fig.add_trace(
        go.Scatter(
            x=np.cos(th), y=np.sin(th), mode="lines", name="unit circle", opacity=0.35
        )
    )
    # 기준 전압(실수축)
    fig.add_trace(
        go.Scatter(x=[0, 1], y=[0, 0], mode="lines+markers", name="V(참조)")
    )
    # 전류(위상 이동)
    fig.add_trace(
        go.Scatter(
            x=[0, math.cos(phi)], y=[0, math.sin(phi)], mode="lines+markers", name="I(위상 이동)"
        )
    )
    lim = 1.3
    fig.update_xaxes(range=[-lim, lim], zeroline=True)
    fig.update_yaxes(range=[-lim, lim], scaleanchor="x", scaleratio=1)
    fig.update_layout(template="plotly_white", title=title, height=420)
    return fig


def waveform_fig(Vp: float, Ip: float, f: float, phi: float, dur: float = 0.1):
    """
    시간영역 파형 v(t), i(t), p(t)=v·i 를 한 Figure에 그리기
    - Vp, Ip: 전압/전류 피크값
    - f: 주파수(Hz)
    - phi: 전압→전류 위상차(rad)
    - dur: 표시 구간(초)
    """
    t = np.linspace(0, dur, 1000)
    v = Vp * np.sin(2 * np.pi * f * t)
    i = Ip * np.sin(2 * np.pi * f * t + phi)
    p = v * i

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=v, name="v(t)"))
    fig.add_trace(go.Scatter(x=t, y=i, name="i(t)"))
    fig.add_trace(go.Scatter(x=t, y=p, name="p(t)=v·i", opacity=0.5))
    fig.update_layout(
        template="plotly_white", height=420, title="시간영역 파형", xaxis_title="t (s)"
    )
    return fig


# ---------------------------------------------------------------------
# 병렬 저항 애니메이션 (점 이동 속도가 전류 비례)
# ---------------------------------------------------------------------
def make_parallel_animation(Rs, V, seconds: int = 5, fps: int = 30):
    """
    병렬 저항 회로를 간단히 배선도로로 나타내고, 가지 전류에 비례한 속도로
    작은 점을 좌→우로 이동시키는 애니메이션 Figure 를 생성합니다.

    Parameters
    ----------
    Rs : list[float]      병렬 저항들(Ω, 양수만 사용)
    V  : float            공급 전압(V, DC)
    seconds : int         애니메이션 길이(초)
    fps : int             프레임/초
    """
    Rs = [float(r) for r in Rs if float(r) > 0]
    N = len(Rs)
    if N == 0:
        return go.Figure()

    # 전류/합성저항
    I_each = [V / r for r in Rs]
    I_tot = float(sum(I_each))
    Req = 1.0 / float(sum(1.0 / r for r in Rs))

    # 배선 좌표
    xL, xR = 1.0, 9.0
    x1, x2 = 2.0, 8.0
    ys = np.linspace(N - 1, 0, N)  # 위에서 아래로 가지 배치
    ylim = (-0.8, N - 0.2)

    fig = go.Figure()

    # 좌/우 버스바(정지 trace #1~2)
    fig.add_trace(go.Scatter(x=[xL, xL], y=[ylim[0], ylim[1]], mode="lines",
                             line=dict(width=3), name="Bus(+)", showlegend=False))
    fig.add_trace(go.Scatter(x=[xR, xR], y=[ylim[0], ylim[1]], mode="lines",
                             line=dict(width=3), name="Bus(-)", showlegend=False))

    # 가지 배선(정지 trace #3~(2+N)): 선 두께 = 전류 비례
    Imax = max(I_each) if I_each else 1.0
    for k, y in enumerate(ys):
        lw = 2 + 6 * (I_each[k] / Imax)
        fig.add_trace(
            go.Scatter(
                x=[xL, x1, x2, xR], y=[y, y, y, y], mode="lines",
                line=dict(width=lw), name=f"R{k+1}", showlegend=False
            )
        )

    # 움직이는 점(마커) trace (마지막 N개: (2+N+1) ~ (2+N+N))
    for k, y in enumerate(ys):
        fig.add_trace(
            go.Scatter(x=[x1], y=[y], mode="markers",
                       marker=dict(size=10), name=f"I{k+1}", showlegend=False)
        )

    # 프레임: 앞의 (2+N)개는 placeholder, 마지막 N개(점)만 위치 갱신
    T = int(seconds * fps)
    speeds = [max(0.05, i / Imax) for i in I_each]  # 최소속도 보호
    frames = []
    for t in range(T):
        data = [go.Scatter() for _ in range(2 + N)]  # 버스바+가지 자리맞춤
        for k, y in enumerate(ys):
            s = ((t / fps) * speeds[k]) % 1.0
            x = x1 + (x2 - x1) * s
            data.append(go.Scatter(x=[x], y=[y]))     # 점 위치만 업데이트
        frames.append(go.Frame(data=data, name=f"f{t}"))
    fig.frames = frames

    # 레이아웃/컨트롤
    fig.update_layout(
        updatemenus=[{
            "type": "buttons",
            "buttons": [
                {"label": "▶ Play", "method": "animate",
                 "args": [None, {"frame": {"duration": int(1000 / fps), "redraw": True},
                                 "fromcurrent": True, "transition": {"duration": 0}}]},
                {"label": "⏸ Pause", "method": "animate",
                 "args": [[None], {"mode": "immediate",
                                   "frame": {"duration": 0, "redraw": False},
                                   "transition": {"duration": 0}}]},
            ],
            "direction": "left", "x": 0.0, "y": 1.08,
        }],
        sliders=[{
            "steps": [{"args": [[f"f{t}"],
                                 {"frame": {"duration": 0, "redraw": True},
                                  "mode": "immediate",
                                  "transition": {"duration": 0}}],
                       "label": f"{t}", "method": "animate"} for t in range(T)],
            "x": 0.05, "y": 1.02, "len": 0.9
        }],
        xaxis=dict(range=[0, 10], zeroline=False, showgrid=False),
        yaxis=dict(range=list(ylim), zeroline=False, showgrid=False,
                   scaleanchor="x", scaleratio=1),
        template="plotly_white",
        margin=dict(l=20, r=20, t=60, b=20),
        title=f"병렬 저항 애니메이션 (V = {V} V,  Req = {Req:.3f} Ω,  Itot = {I_tot:.3f} A)",
        showlegend=False,
    )

    # 가지 라벨(저항/전류)
    annotations = []
    for k, y in enumerate(ys):
        annotations += [
            dict(x=(x1 + x2) / 2, y=y + 0.22, text=f"R{k+1} = {Rs[k]:.3f} Ω",
                 showarrow=False, font=dict(size=12)),
            dict(x=(x1 + x2) / 2, y=y - 0.22, text=f"I{k+1} = {I_each[k]:.3f} A",
                 showarrow=False, font=dict(size=12, color="#1f77b4")),
        ]
    fig.update_layout(annotations=annotations)
    return fig
