# tabs/tab_fft.py
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from utils.plot import line_fig

def _make_window(kind: str, N: int):
    if kind == "Rect (no window)":
        return np.ones(N)
    if kind == "Hann":
        return np.hanning(N)
    if kind == "Hamming":
        return np.hamming(N)
    if kind == "Blackman":
        return np.blackman(N)
    return np.ones(N)

def render():
    st.subheader("푸리에 변환 (DFT/FFT, 창·제로패딩·dB/위상)")

    c1, c2, c3 = st.columns(3)
    with c1:
        fs  = st.slider("샘플링 주파수 fs (Hz)", 64, 4096, 1024, 64)
        dur = st.slider("신호 길이 (초)", 0.25, 5.0, 1.0, 0.25)
    with c2:
        f1 = st.slider("주파수 f1 (Hz)", 1.0, 200.0, 20.0, 1.0)
        A1 = st.slider("진폭 A1", 0.0, 5.0, 1.0, 0.1)
    with c3:
        f2 = st.slider("주파수 f2 (Hz)", 1.0, 200.0, 55.0, 1.0)
        A2 = st.slider("진폭 A2", 0.0, 5.0, 0.8, 0.1)

    c4, c5, c6, c7 = st.columns(4)
    with c4:
        win_kind = st.selectbox("창 함수", ["Rect (no window)", "Hann", "Hamming", "Blackman"])
    with c5:
        zp_factor = st.selectbox("제로패딩 배수", [1, 2, 4, 8], index=1)
    with c6:
        use_db = st.checkbox("dB 스케일", value=False)
    with c7:
        show_phase = st.checkbox("위상 스펙트럼 보기", value=True)

    # ----- 시간영역 신호 -----
    t = np.arange(0, dur, 1/fs)
    x = A1*np.sin(2*np.pi*f1*t) + A2*np.sin(2*np.pi*f2*t)

    # ----- 창 & 제로패딩 -----
    N = len(x)
    w = _make_window(win_kind, N)
    xw = x * w

    Np = int(N * zp_factor)                  # FFT 길이
    X = np.fft.rfft(xw, n=Np)                # 한쪽 스펙트럼
    freqs = np.fft.rfftfreq(Np, d=1/fs)

    # ----- 진폭 스케일(창 보정 포함) -----
    # 한쪽 스펙트럼의 단일 사인에서, 피크의 |X_k| ≈ (A/2)*sum(w)
    # => A ≈ 2|X_k| / sum(w). (DC/나이퀴스트는 2배 보정 제외)
    mag = (2.0 / np.sum(w)) * np.abs(X)
    if mag.size > 0:
        mag[0] /= 2.0
        if (Np % 2 == 0) and (mag.size >= 2):
            mag[-1] /= 2.0

    # dB 변환(작은 수치 안정화)
    if use_db:
        mag_plot = 20*np.log10(np.maximum(mag, 1e-12))
        ylab = "Magnitude (dBFS, 창보정)"
    else:
        mag_plot = mag
        ylab = "Amplitude (창보정)"

    # ----- 시간/주파수 그래프 -----
    fig_time = line_fig(t, [x, xw], ["x(t)", f"x(t)×{win_kind}"], "시간 영역", "t (s)", "amplitude")

    fig_mag = go.Figure()
    fig_mag.add_trace(go.Scatter(x=freqs, y=mag_plot, mode="lines", name="|X(f)|"))
    fig_mag.update_layout(
        title="주파수 영역 (진폭 스펙트럼)",
        xaxis_title="f (Hz)", yaxis_title=ylab, template="plotly_white", height=420
    )

    p1, p2 = st.columns(2)
    with p1: st.plotly_chart(fig_time, use_container_width=True)
    with p2: st.plotly_chart(fig_mag, use_container_width=True)

    # ----- 위상 스펙트럼(옵션) -----
    if show_phase:
        # 너무 작은 구간은 위상 노이즈 → 마스크
        mask = mag > (mag.max()*1e-3)
        phase = np.angle(X, deg=True)
        phase_plot = np.where(mask, phase, np.nan)

        fig_phase = go.Figure()
        fig_phase.add_trace(go.Scatter(x=freqs, y=phase_plot, mode="lines", name="∠X(f)"))
        fig_phase.update_layout(
            title="위상 스펙트럼 (deg)", xaxis_title="f (Hz)",
            yaxis_title="phase (°)", template="plotly_white", height=320
        )
        st.plotly_chart(fig_phase, use_container_width=True)

    # 참고 메시지
    st.caption(
        "창을 쓰면 leakage가 줄지만 **진폭이 줄어드는 만큼 보정**(sum(w) 사용)을 적용했습니다. "
        "제로패딩은 분해능을 높여 피크 주파수를 더 정밀하게 읽도록 도와주지만, 실제 정보량을 늘리진 않습니다."
    )
