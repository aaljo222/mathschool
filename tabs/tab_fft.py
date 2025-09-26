# tabs/tab_fft.py
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from utils.plot import line_fig

def render():
    st.subheader("푸리에 변환 (이산 푸리에 변환, FFT)")
    c1, c2, c3 = st.columns(3)
    with c1:
        fs = st.slider("샘플링 주파수 fs", 64, 4096, 1024, 64)
        dur = st.slider("신호 길이 (초)", 0.25, 5.0, 1.0, 0.25)
    with c2:
        f1 = st.slider("주파수 f1", 1.0, 100.0, 10.0, 1.0)
        A1 = st.slider("진폭 A1", 0.0, 5.0, 1.0, 0.1)
    with c3:
        f2 = st.slider("주파수 f2", 1.0, 100.0, 25.0, 1.0)
        A2 = st.slider("진폭 A2", 0.0, 5.0, 0.7, 0.1)

    t = np.arange(0, dur, 1/fs)
    x = A1*np.sin(2*np.pi*f1*t) + A2*np.sin(2*np.pi*f2*t)
    N = len(x); X = np.fft.rfft(x); freqs = np.fft.rfftfreq(N, d=1/fs); mag = np.abs(X)*(2/N)

    fig_time = line_fig(t, [x], ["signal"], "시간 영역", "t (s)", "x(t)")
    fig_freq = go.Figure(); fig_freq.add_trace(go.Bar(x=freqs, y=mag, name="|X(f)|"))
    fig_freq.update_layout(title="주파수 영역 (진폭 스펙트럼)", xaxis_title="f (Hz)", yaxis_title="Magnitude", template="plotly_white")

    p1, p2 = st.columns(2)
    with p1: st.plotly_chart(fig_time, use_container_width=True)
    with p2: st.plotly_chart(fig_freq, use_container_width=True)
