# app.py
import math
import time
import numpy as np
import streamlit as st
import plotly.graph_objs as go
import matplotlib.animation as animation
import matplotlib.pyplot as plt


# ───── 페이지 설정 ─────
st.set_page_config(page_title="수학 애니메이션 튜터", layout="wide")

# ───── 최초 1회 공지 ─────
if "show_notice" not in st.session_state:
    st.session_state.show_notice = True   # 첫 방문에만 보여주기

NOTICE_MD = """
### ✨ 업데이트 안내
- 이 앱은 **매주 새로운 수학 애니메이션**을 추가합니다.
- 현재는 **중·고등학교 수학**(포물선/쌍곡선, 삼각함수, 미분·적분, 선형회귀, 테일러 시리즈, 푸리에 변환) 위주로 제공됩니다.

### 📬 교육 관계자 연락처
👉 **[aaljo2@naver.com](mailto:aaljo2@naver.com)**  
**교육 콘텐츠 개발**·맞춤 커리큘럼 제작을 도와드립니다.
"""

def render_notice_body():
    st.markdown(NOTICE_MD)
    st.divider()
    if st.button("닫기", key="notice_close_btn", use_container_width=True):
        st.session_state.show_notice = False
        try:
            st.rerun()                 # 최신 버전
        except Exception:
            st.experimental_rerun()    # 구버전 대응

if st.session_state.show_notice:
    if hasattr(st, "dialog"):          # Streamlit ≥ 1.36
        @st.dialog("📢 공지사항")
        def _notice_dialog():
            render_notice_body()
        _notice_dialog()
    elif hasattr(st, "modal"):         # 1.32 ~ 1.35
        with st.modal("📢 공지사항"):
            render_notice_body()
    else:                              # 더 구버전
        with st.expander("📢 공지사항", expanded=True):
            render_notice_body()

# ───── 앱 제목 ─────
st.title("수학 애니메이션 튜터 (Streamlit, Free Plan)")

# ───── 탭 구성 ─────
tabs = st.tabs([
    "포물선/쌍곡선", "삼각함수", "미분·적분(정의)",
    "선형회귀", "테일러 시리즈", "푸리에 변환",
    "오일러 공식(애니메이션)", "벡터의 선형결합"
])
# --------------------- 공통 유틸 ---------------------
def line_fig(x, ys, names, title, xaxis="x", yaxis="y"):
    fig = go.Figure()
    for y, name in zip(ys, names):
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name))
    fig.update_layout(title=title, xaxis_title=xaxis, yaxis_title=yaxis, template="plotly_white")
    return fig

def contour_implicit(F, x_range, y_range, level=0.0, title=""):
    xs = np.linspace(*x_range, 600)
    ys = np.linspace(*y_range, 600)
    X, Y = np.meshgrid(xs, ys)
    Z = F(X, Y)
    fig = go.Figure(
        data=go.Contour(
            x=xs, y=ys, z=Z,
            contours=dict(start=level, end=level, size=1, coloring="none"),
            line_width=2, showscale=False
        )
    )
    fig.update_layout(title=title, xaxis_title="x", yaxis_title="y", template="plotly_white")
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig

# --------------------- 1) 포물선 / 쌍곡선 ---------------------
with tabs[0]:
    st.subheader("포물선 / 쌍곡선 시각화 (Implicit Contour)")
    shape = st.radio("곡선 선택", ["포물선 (Parabola)", "쌍곡선 (Hyperbola)"], horizontal=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        h = st.slider("중심 h", -5.0, 5.0, 0.0, 0.1)
        k = st.slider("중심 k", -5.0, 5.0, 0.0, 0.1)
    with col2:
        a = st.slider("a", 0.5, 5.0, 2.0, 0.1)
        b = st.slider("b (쌍곡선에서 사용)", 0.5, 5.0, 1.5, 0.1)
    with col3:
        x_min, x_max = st.slider("x-범위", -10, 10, (-6, 6))
        y_min, y_max = st.slider("y-범위", -10, 10, (-6, 6))

    if shape.startswith("포물선"):
        F = lambda X, Y: Y - (a*(X - h)**2 + k)
        fig = contour_implicit(F, (x_min, x_max), (y_min, y_max), title="포물선")
    else:
        F = lambda X, Y: ((X - h)**2)/(a**2) - ((Y - k)**2)/(b**2) - 1
        fig = contour_implicit(F, (x_min, x_max), (y_min, y_max), title="쌍곡선")

    st.plotly_chart(fig, use_container_width=True)

# --------------------- 2) 삼각함수 ---------------------
with tabs[1]:
    st.subheader("삼각함수: sin, cos (주파수/위상 조절)")
    col1, col2, col3 = st.columns(3)
    with col1: f = st.slider("주파수 f (Hz)", 0.1, 5.0, 1.0, 0.1)
    with col2: A = st.slider("진폭 A", 0.5, 3.0, 1.0, 0.1)
    with col3: phi = st.slider("위상 φ (라디안)", -np.pi, np.pi, 0.0, 0.1)

    t = np.linspace(0, 2, 1000)
    y_sin = A*np.sin(2*np.pi*f*t + phi)
    y_cos = A*np.cos(2*np.pi*f*t + phi)
    fig = line_fig(t, [y_sin, y_cos], ["A·sin(2πft+φ)", "A·cos(2πft+φ)"], "삼각함수")
    st.plotly_chart(fig, use_container_width=True)

# --------------------- 3) 미분·적분 (정의 기반) ---------------------
with tabs[2]:
    st.subheader("미분·적분 (정의)")
    funcs = {
        "x^2": (lambda x: x**2, lambda x: 2*x, lambda a,b: (b**3 - a**3)/3),
        "3x^3 + 6x": (lambda x: 3*x**3 + 6*x, lambda x: 9*x**2 + 6, lambda a,b: (3/4)*(b**4-a**4)+3*(b**2-a**2)),
        "sin x": (lambda x: np.sin(x), lambda x: np.cos(x), lambda a,b: -np.cos(b)+np.cos(a)),
        "e^x": (lambda x: np.exp(x), lambda x: np.exp(x), lambda a,b: np.exp(b)-np.exp(a)),
    }
    f_name = st.selectbox("함수 선택", list(funcs.keys()), index=1)
    f, fprime, Fint = funcs[f_name]

    st.markdown("### 미분 (극한 정의)  $\\lim_{h\\to 0}\\frac{f(a+h)-f(a)}{h}$")
    col1, col2 = st.columns(2)
    with col1:
        a0 = st.slider("미분 지점 a", -3.0, 3.0, 1.0, 0.1)
        h = st.slider("증분 h", 1e-5, 0.5, 0.1, 0.0001, format="%.5f")
        deriv_est = (f(a0+h) - f(a0)) / h
        deriv_true = fprime(a0)
        st.write(f"수치 미분 ≈ **{deriv_est:.6f}**,  해석 미분 = **{deriv_true:.6f}**")
    with col2:
        xs = np.linspace(-3, 3, 800)
        ys = f(xs)
        tangent = f(a0) + deriv_true*(xs-a0)
        fig_d = line_fig(xs, [ys, tangent], [f_name, "접선(해석 기울기)"], "함수와 접선")
        fig_d.add_trace(go.Scatter(x=[a0], y=[f(a0)], mode="markers", name="a"))
        st.plotly_chart(fig_d, use_container_width=True)

    st.markdown("### 정적분 (리만 합)  $\\int_a^b f(x)\\,dx$")
    col3, col4 = st.columns(2)
    with col3:
        A_int, B_int = st.slider("구간 [a, b]", -3.0, 3.0, (-1.0, 2.0), 0.1)
        N = st.slider("직사각형 개수 N", 2, 400, 30, 2)
        xs = np.linspace(A_int, B_int, N+1)
        mids = (xs[:-1] + xs[1:]) / 2
        dx = (B_int - A_int)/N
        approx = np.sum(f(mids) * dx)
        exact = Fint(A_int, B_int)
        st.write(f"리만합 ≈ **{approx:.6f}**,  해석적 값 = **{exact:.6f}**,  오차 = **{approx-exact:.6e}**")
    with col4:
        X = np.linspace(A_int, B_int, 1000)
        Y = f(X)
        fig_i = go.Figure()
        fig_i.add_trace(go.Scatter(x=X, y=Y, mode="lines", name=f_name))
        for i in range(N):
            x0, x1 = xs[i], xs[i+1]
            xm = (x0+x1)/2
            y = f(xm)
            fig_i.add_shape(
                type="rect",
                x0=x0, x1=x1, y0=0, y1=y,
                line=dict(width=1),
                fillcolor="LightSkyBlue",
                opacity=0.2
            )
        fig_i.update_layout(title="리만 합(중점 규칙)", template="plotly_white")
        st.plotly_chart(fig_i, use_container_width=True)

# --------------------- 4) 선형회귀 ---------------------
with tabs[3]:
    st.subheader("선형회귀: y = ax + b (경사하강)")
    col1, col2, col3 = st.columns(3)
    with col1:
        true_a = st.slider("진짜 기울기 (데이터 생성)", -5.0, 5.0, 2.0, 0.1)
        true_b = st.slider("진짜 절편 (데이터 생성)", -10.0, 10.0, -1.0, 0.1)
    with col2:
        noise = st.slider("노이즈 표준편차", 0.0, 5.0, 1.0, 0.1)
        npts = st.slider("데이터 개수", 10, 300, 80, 10)
    with col3:
        lr = st.slider("학습률(learning rate)", 1e-5, 0.5, 0.05, 0.0005, format="%.5f")
        steps = st.slider("경사하강 스텝 수", 1, 2000, 200, 10)

    rng = np.random.default_rng(0)
    x = np.linspace(-5, 5, npts)
    y = true_a*x + true_b + rng.normal(0, noise, size=npts)

    def mse(a, b): return np.mean((y - (a*x + b))**2)
    def grad(a, b):
        n = len(x); e = (a*x + b) - y
        return (1.0/n)*np.sum(e*x), (1.0/n)*np.sum(e)

    a_hat, b_hat = 0.0, 0.0
    history = []
    for _ in range(steps):
        da, db = grad(a_hat, b_hat)
        a_hat -= lr*da; b_hat -= lr*db
        history.append(mse(a_hat, b_hat))

    fig_lr = go.Figure()
    fig_lr.add_trace(go.Scatter(x=x, y=y, mode="markers", name="data"))
    fig_lr.add_trace(go.Scatter(x=x, y=true_a*x+true_b, mode="lines", name="true line"))
    fig_lr.add_trace(go.Scatter(x=x, y=a_hat*x+b_hat, mode="lines", name="learned line"))
    fig_lr.update_layout(title=f"학습 결과: a≈{a_hat:.3f}, b≈{b_hat:.3f}", template="plotly_white")

    fig_mse = go.Figure()
    fig_mse.add_trace(go.Scatter(x=np.arange(1, steps+1), y=history, mode="lines", name="MSE"))
    fig_mse.update_layout(title="MSE 감소", xaxis_title="step", yaxis_title="MSE", template="plotly_white")

    colp1, colp2 = st.columns(2)
    with colp1: st.plotly_chart(fig_lr, use_container_width=True)
    with colp2: st.plotly_chart(fig_mse, use_container_width=True)

# --------------------- 5) 테일러 시리즈 ---------------------
with tabs[4]:
    st.subheader("테일러 시리즈 근사")
    funcs_T = {"sin x": (lambda x: np.sin(x), "sin"),
               "cos x": (lambda x: np.cos(x), "cos"),
               "e^x":   (lambda x: np.exp(x), "exp")}
    fname = st.selectbox("함수 선택", list(funcs_T.keys()))
    f_true, kind = funcs_T[fname]
    c = st.slider("전개 중심 c", -2.0, 2.0, 0.0, 0.1)
    order = st.slider("차수 N", 0, 20, 6, 1)
    x_min, x_max = st.slider("x-범위", -10.0, 10.0, (-6.0, 6.0), 0.5)
    X = np.linspace(x_min, x_max, 1200)

    def taylor_series(kind, X, c, N):
        Ts = np.zeros_like(X)
        if kind == "exp":
            for n in range(N+1): Ts += (math.exp(c)/math.factorial(n)) * (X - c)**n
        elif kind == "sin":
            derivs = [np.sin, np.cos, lambda z:-np.sin(z), lambda z:-np.cos(z)]
            for n in range(N+1):
                Ts += (derivs[n%4](c) / math.factorial(n)) * (X - c)**n
        elif kind == "cos":
            derivs = [np.cos, lambda z:-np.sin(z), lambda z:-np.cos(z), np.sin]
            for n in range(N+1):
                Ts += (derivs[n%4](c) / math.factorial(n)) * (X - c)**n
        return Ts

    T = taylor_series(kind, X, c, order)
    fig_T = line_fig(X, [f_true(X), T], ["원함수", f"테일러 근사 (N={order})"], f"{fname} Taylor around c={c}")
    st.plotly_chart(fig_T, use_container_width=True)

# --------------------- 6) 푸리에 변환 (FFT) ---------------------
with tabs[5]:
    st.subheader("푸리에 변환 (이산 푸리에 변환, FFT)")
    col1, col2, col3 = st.columns(3)
    with col1:
        fs = st.slider("샘플링 주파수 fs", 64, 4096, 1024, 64)
        dur = st.slider("신호 길이 (초)", 0.25, 5.0, 1.0, 0.25)
    with col2:
        f1 = st.slider("주파수 f1", 1.0, 100.0, 10.0, 1.0)
        A1 = st.slider("진폭 A1", 0.0, 5.0, 1.0, 0.1)
    with col3:
        f2 = st.slider("주파수 f2", 1.0, 100.0, 25.0, 1.0)
        A2 = st.slider("진폭 A2", 0.0, 5.0, 0.7, 0.1)

    t = np.arange(0, dur, 1/fs)
    x = A1*np.sin(2*np.pi*f1*t) + A2*np.sin(2*np.pi*f2*t)
    N = len(x); X = np.fft.rfft(x); freqs = np.fft.rfftfreq(N, d=1/fs); mag = np.abs(X)*(2/N)

    fig_time = line_fig(t, [x], ["signal"], "시간 영역", "t (s)", "x(t)")
    fig_freq = go.Figure(); fig_freq.add_trace(go.Bar(x=freqs, y=mag, name="|X(f)|"))
    fig_freq.update_layout(title="주파수 영역 (진폭 스펙트럼)", xaxis_title="f (Hz)", yaxis_title="Magnitude", template="plotly_white")

    colp1, colp2 = st.columns(2)
    with colp1: st.plotly_chart(fig_time, use_container_width=True)
    with colp2: st.plotly_chart(fig_freq, use_container_width=True)

# --------------------- 7) 오일러 공식(애니메이션) ---------------------
with tabs[6]:
    st.subheader("오일러 공식  $e^{i\\omega t} = \\cos(\\omega t) + i\\sin(\\omega t)$  애니메이션")

    # 컨트롤
    c1, c2, c3, c4 = st.columns(4)
    with c1: freq = st.slider("주파수 f (Hz)", 0.1, 5.0, 1.0, 0.1, key="e_freq")
    with c2: amp  = st.slider("진폭 A", 0.5, 2.0, 1.0, 0.1, key="e_amp")
    with c3: secs = st.slider("재생 길이(초)", 1, 10, 5, 1, key="e_secs")
    with c4: fps  = st.slider("FPS", 5, 40, 20, 1, key="e_fps")

    omega = 2*np.pi*freq
    total_frames = int(secs*fps)

    if "euler_play" not in st.session_state: st.session_state.euler_play = False
    b1, b2, _ = st.columns([1,1,6])
    with b1:
        if st.button("▶ 재생", key="e_play"): st.session_state.euler_play = True
    with b2:
        if st.button("⏹ 정지", key="e_stop"): st.session_state.euler_play = False

    # 좌/우 출력 플레이스홀더 (여기서 key 사용 X)
    left, right = st.columns(2)
    with left:  ph_circle = st.empty()
    with right: ph_wave   = st.empty()

    def circle_fig(x, y):
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

    def wave_fig(t_axis, y_axis, t_now, y_now):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t_axis, y=y_axis, mode="lines", name="sin(ωt)"))
        fig.add_trace(go.Scatter(x=[t_now], y=[y_now], mode="markers", name="현재"))
        fig.add_vline(x=t_now, line_width=1, line_dash="dot")
        fig.update_layout(template="plotly_white", height=480, title="실수부(우): sin(ωt)",
                          xaxis_title="t (s)", yaxis_title="Amplitude")
        return fig

    if st.session_state.euler_play:
        start = time.perf_counter()
        t_hist, y_hist = [], []
        for frame in range(total_frames):
            if not st.session_state.euler_play: break
            t = frame / fps
            x = amp*np.cos(omega*t); y = amp*np.sin(omega*t)
            t_hist.append(t); y_hist.append(np.sin(omega*t))

            ph_circle.plotly_chart(circle_fig(x, y), use_container_width=True)
            ph_wave.plotly_chart(wave_fig(np.array(t_hist), np.array(y_hist), t, y_hist[-1]),
                                 use_container_width=True)

            sleep = (frame+1)/fps - (time.perf_counter() - start)
            if sleep > 0: time.sleep(sleep)
        st.session_state.euler_play = False
    else:
        ph_circle.plotly_chart(circle_fig(amp*np.cos(0), amp*np.sin(0)), use_container_width=True)
        ph_wave.plotly_chart(wave_fig(np.array([0.0]), np.array([0.0]), 0.0, 0.0), use_container_width=True)
# ───── 벡터 선형결합 탭 ─────
with tabs[7]:
    st.header("벡터의 선형결합 애니메이션")

    # 두 개의 기본 벡터 정의
    v1 = np.array([2, 1])
    v2 = np.array([1, 2])

    fig, ax = plt.subplots()
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 6)
    ax.grid(True)
    ax.set_aspect('equal')

    # 기본 벡터 그리기
    ax.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, color='r', label='v1')
    ax.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1, color='b', label='v2')

    result_arrow = ax.quiver(0, 0, 0, 0, angles='xy', scale_units='xy', scale=1, color='g', label='a*v1 + b*v2')

    ax.legend()

    # 애니메이션 업데이트 함수
    def update(frame):
        a = np.cos(frame/20)
        b = np.sin(frame/20)
        result = a*v1 + b*v2
        result_arrow.set_UVC(result[0], result[1])
        return result_arrow,

    ani = animation.FuncAnimation(fig, update, frames=200, interval=100, blit=True)

    st.pyplot(fig)
st.markdown("---")
st.caption("이재오에게 저작권이 있으며 개발이나 협업하고자 하시는 관계자는 연락바랍니다")
