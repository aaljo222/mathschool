# tabs/tab_euler.py
from pathlib import Path
import time, math
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from utils.media import show_gif_cached, show_image, WIX_HEADERS  # ← 추가

# Euler 탭 상단에 보여줄 전용 GIF/이미지 URL (가능하면 토큰 없는 안정 URL 권장)
EULER_GIF_URL = (
    "https://images-wixmp-ed30a86b8c4ca887773594c2.wixmp.com/f/dcbf3a55-6bea-4cfe-ac74-d3754be91a8e/d4g48y9-4aaf877b-ad4d-4c2c-8483-254ba4c1d9f3.gif?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1cm46YXBwOjdlMGQxODg5ODIyNjQzNzNhNWYwZDQxNWVhMGQyNmUwIiwiaXNzIjoidXJuOmFwcDo3ZTBkMTg4OTgyMjY0MzczYTVmMGQ0MTVlYTBkMjZlMCIsIm9iaiI6W1t7InBhdGgiOiIvZi9kY2JmM2E1NS02YmVhLTRjZmUtYWM3NC1kMzc1NGJlOTFhOGUvZDRnNDh5OS00YWFmODc3Yi1hZDRkLTRjMmMtODQ4My0yNTRiYTRjMWQ5ZjMuZ2lmIn1dXSwiYXVkIjpbInVybjpzZXJ2aWNlOmZpbGUuZG93bmxvYWQiXX0.Wy9lJaw2WPGeNzu8fWls_mlBIjIE3bu5WGo6DVMi358"
)

def _circle_fig(amp, x, y):
    th = np.linspace(0, 2*np.pi, 400)
    cx, cy = amp*np.cos(th), amp*np.sin(th)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cx, y=cy, mode="lines", name="원", opacity=0.45))
    fig.add_trace(go.Scatter(x=[0, x], y=[0, y], mode="lines+markers", name="$e^{i\\omega t}$"))
    fig.add_trace(go.Scatter(x=[x, x], y=[0, y], mode="lines", line=dict(dash="dot"), showlegend=False))
    fig.add_trace(go.Scatter(x=[0, x], y=[y, y], mode="lines", line=dict(dash="dot"), showlegend=False))
    lim = amp*1.2
    fig.update_xaxes(range=[-lim, lim]); fig.update_yaxes(range=[-lim, lim], scaleanchor="x", scaleratio=1)
    fig.update_layout(template="plotly_white", height=480, title="복소평면(좌): 원 위 회전")
    return fig

def _wave_fig(t_axis, y_axis, t_now, y_now):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_axis, y=y_axis, mode="lines", name="sin(ωt)"))
    fig.add_trace(go.Scatter(x=[t_now], y=[y_now], mode="markers", name="현재"))
    fig.add_vline(x=t_now, line_width=1, line_dash="dot")
    fig.update_layout(template="plotly_white", height=480, title="실수부(우): sin(ωt)",
                      xaxis_title="t (s)", yaxis_title="Amplitude")
    return fig

def render():
    # ▶ Euler 탭 전용 이미지 (다른 탭과 독립)
    show_gif_cached(
        EULER_GIF_URL,
        filename="euler_demo.gif",    # 캐시 파일명
        caption="오일러 공식 데모(GIF, 캐시)",
        headers=WIX_HEADERS,          # wixmp 차단 우회
        subdir="euler"                # 캐시를 euler 하위 폴더에 보관
    )

    st.subheader("오일러 공식  $e^{i\\omega t} = \\cos(\\omega t) + i\\sin(\\omega t)$  애니메이션")
    c1, c2, c3, c4 = st.columns(4)
    with c1: freq = st.slider("주파수 f (Hz)", 0.1, 5.0, 1.0, 0.1, key="e_freq")
    with c2: amp  = st.slider("진폭 A", 0.5, 2.0, 1.0, 0.1, key="e_amp")
    with c3: secs = st.slider("재생 길이(초)", 1, 10, 5, 1, key="e_secs")
    with c4: fps  = st.slider("FPS", 5, 40, 20, 1, key="e_fps")

    omega = 2*np.pi*freq; total_frames = int(secs*fps)

    if "euler_play" not in st.session_state: st.session_state.euler_play = False
    b1, b2, _ = st.columns([1,1,6])
    with b1:
        if st.button("▶ 재생", key="e_play"): st.session_state.euler_play = True
    with b2:
        if st.button("⏹ 정지", key="e_stop"): st.session_state.euler_play = False

    left, right = st.columns(2)
    with left:  ph_circle = st.empty()
    with right: ph_wave   = st.empty()

    if st.session_state.euler_play:
        start = time.perf_counter(); t_hist, y_hist = [], []
        for frame in range(total_frames):
            if not st.session_state.euler_play: break
            t = frame / fps
            x = amp*np.cos(omega*t); y = amp*np.sin(omega*t)
            t_hist.append(t); y_hist.append(np.sin(omega*t))
            ph_circle.plotly_chart(_circle_fig(amp, x, y), use_container_width=True)
            ph_wave.plotly_chart(_wave_fig(np.array(t_hist), np.array(y_hist), t, y_hist[-1]),
                                 use_container_width=True)
            sleep = (frame+1)/fps - (time.perf_counter() - start)
            if sleep > 0: time.sleep(sleep)
        st.session_state.euler_play = False
    else:
        ph_circle.plotly_chart(_circle_fig(amp, amp*np.cos(0), amp*np.sin(0)), use_container_width=True)
        ph_wave.plotly_chart(_wave_fig(np.array([0.0]), np.array([0.0]), 0.0, 0.0), use_container_width=True)
