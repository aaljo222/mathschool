# tabs/tab_euler.py
from pathlib import Path
import time, math
import numpy as np
import requests
import streamlit as st
import plotly.graph_objects as go

# --- Streamlit 버전 호환용 이미지 출력 래퍼 ---
def show_image(src, caption=None):
    """Streamlit 버전에 따라 image 폭 옵션을 자동 선택"""
    try:
        st.image(src, caption=caption, use_container_width=True)   # 최신
    except TypeError:
        st.image(src, caption=caption, use_column_width=True)      # 구버전

# --- 외부 GIF 로컬 캐시 (wixmp 차단 우회용 헤더 포함) ---
ASSET_DIR = Path("public/assets")
ASSET_DIR.mkdir(parents=True, exist_ok=True)

# 토큰이 붙은 주소는 곧 만료됩니다. 토큰 없이 원본 경로(또는 직접 호스팅한 경로)를 쓰세요.
GIF_URL = (
    "https://images-wixmp-ed30a86b8c4ca887773594c2.wixmp.com/f/dcbf3a55-6bea-4cfe-ac74-d3754be91a8e/d4g48y9-4aaf877b-ad4d-4c2c-8483-254ba4c1d9f3.gif?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1cm46YXBwOjdlMGQxODg5ODIyNjQzNzNhNWYwZDQxNWVhMGQyNmUwIiwiaXNzIjoidXJuOmFwcDo3ZTBkMTg4OTgyMjY0MzczYTVmMGQ0MTVlYTBkMjZlMCIsIm9iaiI6W1t7InBhdGgiOiIvZi9kY2JmM2E1NS02YmVhLTRjZmUtYWM3NC1kMzc1NGJlOTFhOGUvZDRnNDh5OS00YWFmODc3Yi1hZDRkLTRjMmMtODQ4My0yNTRiYTRjMWQ5ZjMuZ2lmIn1dXSwiYXVkIjpbInVybjpzZXJ2aWNlOmZpbGUuZG93bmxvYWQiXX0.Wy9lJaw2WPGeNzu8fWls_mlBIjIE3bu5WGo6DVMi358"
)

WIX_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://www.deviantart.com/",
}

def try_cache_remote(url: str, filename: str = "euler_demo.gif") -> Path:
    out_path = ASSET_DIR / filename
    if out_path.exists():
        return out_path
    r = requests.get(url, headers=WIX_HEADERS, timeout=(5, 20))
    if r.status_code in (401, 403):
        raise RuntimeError(f"blocked_by_host ({r.status_code})")
    r.raise_for_status()
    out_path.write_bytes(r.content)
    return out_path

# --- Plotly 도우미 ---
def _circle_fig(amp, x, y):
    th = np.linspace(0, 2*np.pi, 400)
    cx, cy = amp*np.cos(th), amp*np.sin(th)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cx, y=cy, mode="lines", name="원", opacity=0.45))
    fig.add_trace(go.Scatter(x=[0, x], y=[0, y], mode="lines+markers", name="$e^{i\\omega t}$"))
    fig.add_trace(go.Scatter(x=[x, x], y=[0, y], mode="lines", line=dict(dash="dot"), showlegend=False))
    fig.add_trace(go.Scatter(x=[0, x], y=[y, y], mode="lines", line=dict(dash="dot"), showlegend=False))
    lim = amp*1.2
    fig.update_xaxes(range=[-lim, lim])
    fig.update_yaxes(range=[-lim, lim], scaleanchor="x", scaleratio=1)
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

# --- Euler 탭 ---
def render():
    # (선택) 상단 GIF 보여주기: 캐시 성공 → 로컬, 실패 → 링크만
    try:
        local_gif = try_cache_remote(GIF_URL)
        show_image(str(local_gif), caption="외부 GIF(로컬 캐시)")
    except Exception as e:
        st.info("호스팅 제한으로 직접 열어야 할 수 있어요.")
        st.markdown(f"[GIF 새 탭에서 열기]({GIF_URL})")

    st.subheader("오일러 공식  $e^{i\\omega t} = \\cos(\\omega t) + i\\sin(\\omega t)$  애니메이션")
    c1, c2, c3, c4 = st.columns(4)
    with c1: freq = st.slider("주파수 f (Hz)", 0.1, 5.0, 1.0, 0.1, key="e_freq")
    with c2: amp  = st.slider("진폭 A", 0.5, 2.0, 1.0, 0.1, key="e_amp")
    with c3: secs = st.slider("재생 길이(초)", 1, 10, 5, 1, key="e_secs")
    with c4: fps  = st.slider("FPS", 5, 40, 20, 1, key="e_fps")

    omega = 2*np.pi*freq
    total_frames = int(secs*fps)

    if "euler_play" not in st.session_state:
        st.session_state.euler_play = False

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
