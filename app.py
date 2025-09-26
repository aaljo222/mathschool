import math, time
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# ── Modal helpers ─────────────────────────────────────────────────────────────
def open_modal(title: str, body_md: str, key: str = "evt"):
    """Streamlit 버전에 맞춰 dialog/modal로 열기"""
    def _content():
        st.markdown(body_md)
        c1, c2 = st.columns([1,1])
        if c1.button("닫기", key=f"{key}_close"):
            st.session_state[key] = False
            try: st.rerun()
            except Exception: st.experimental_rerun()

    if hasattr(st, "dialog"):           # Streamlit >= 1.36
        @st.dialog(title)
        def _dlg(): _content()
        _dlg()
    elif hasattr(st, "modal"):           # 1.32 ~ 1.35
        with st.modal(title): _content()
    else:                                # fallback
        st.warning(body_md)

def trigger_modal(payload: dict):
    """어디서든 이벤트마다 호출 → 전역 핸들러가 처리"""
    st.session_state["__modal_payload__"] = payload

# ---- 유틸: 분수 ----
def _gcd(a:int, b:int)->int:
    a, b = abs(a), abs(b)
    while b: a, b = b, a % b
    return max(a, 1)

def _lcm(a:int, b:int)->int:
    return abs(a*b) // _gcd(a,b)

def simplify(n:int, d:int):
    if d == 0: return n, d
    g = _gcd(n, d)
    n //= g; d //= g
    if d < 0: n, d = -n, -d
    return n, d

def add_fractions(n1,d1,n2,d2):
    L = _lcm(d1, d2)
    n = n1*(L//d1) + n2*(L//d2)
    return simplify(n, L)

def to_mixed(n:int, d:int):
    if d == 0: return None
    q, r = divmod(abs(n), d)
    sgn = -1 if n<0 else 1
    return sgn*q, r, d  # (정수부, 분자, 분모)

# ---- 유틸: Plotly 그리기 ----
def phasor_fig(V, I, phi, title="Phasor"):
    # V: 전압(피크), I: 전류(피크), phi: 위상차(rad, V→I)
    R = max(V, I) * 1.25 + 0.1
    th = np.linspace(0, 2*np.pi, 360)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=R*np.cos(th)/R, y=R*np.sin(th)/R, mode="lines", name="unit circle", opacity=0.35))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 0], mode="lines+markers", name="V(참조)"))
    fig.add_trace(go.Scatter(x=[0, math.cos(phi)], y=[0, math.sin(phi)], mode="lines+markers", name="I(위상 이동)"))
    lim = 1.3
    fig.update_xaxes(range=[-lim, lim], zeroline=True); fig.update_yaxes(range=[-lim, lim], scaleanchor="x", scaleratio=1)
    fig.update_layout(template="plotly_white", title=title, height=420)
    return fig

def waveform_fig(Vp, Ip, f, phi, dur=0.1):
    t = np.linspace(0, dur, 1000)
    v = Vp*np.sin(2*np.pi*f*t)
    i = Ip*np.sin(2*np.pi*f*t + phi)
    p = v*i
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=v, name="v(t)"))
    fig.add_trace(go.Scatter(x=t, y=i, name="i(t)"))
    fig.add_trace(go.Scatter(x=t, y=p, name="p(t)=v·i", opacity=0.5))
    fig.update_layout(template="plotly_white", height=420,
                      title="시간영역 파형", xaxis_title="t (s)")
    return fig

def ohm_dc_result(V=None, I=None, R=None):
    # 세 값 중 1개 비워두면 나머지로 계산
    if V is None: V = I*R
    if I is None: I = V/R if R!=0 else float("inf")
    if R is None: R = V/I if I!=0 else float("inf")
    P = V*I
    return V, I, R, P

def series_parallel_req(values):
    vals = [v for v in values if v>0]
    if not vals: return 0.0, 0.0
    r_series = float(np.sum(vals))
    r_parallel = 1.0 / np.sum([1.0/v for v in vals]) if all(v>0 for v in vals) else float("inf")
    return r_series, r_parallel



# ───── 페이지 설정 ─────
st.set_page_config(page_title="수학 애니메이션 튜터", layout="wide")
# 탭이 많을 때 가로 스크롤 + 2줄까지 줄바꿈
st.markdown("""
<style>
/* 탭 리스트 영역 */
.stTabs [role="tablist"]{
  gap: .25rem;
  overflow-x: auto;         /* 가로 스크롤 */
  padding: .25rem 0;
  scrollbar-width: thin;
  flex-wrap: wrap;          /* 2줄 이상 줄바꿈 허용 */
}
/* 각 탭 버튼 */
.stTabs [role="tab"]{
  flex: 0 0 auto;           /* 줄바꿈/스크롤 시 폭 고정 */
  font-size: .95rem;        /* 글자 조금 줄이기 */
  padding: .35rem .7rem;    /* 패딩 축소 */
}
</style>
""", unsafe_allow_html=True)

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


# ───── 탭 구성 (단 한 번만 선언) ─────
tabs = st.tabs([
    "포물선/쌍곡선", "삼각함수", "미분·적분(정의)",
    "선형회귀", "테일러 시리즈", "푸리에 변환",
    "오일러 공식(애니메이션)", "벡터의 선형결합",
    "기초도구(전기·분수)"          # ← 새로 추가
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
    st.subheader("벡터의 선형결합:  a·v₁ + b·v₂")

    # 입력 UI
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

    # 축 범위
    max_len = max(1.0, float(np.linalg.norm(v1) + np.linalg.norm(v2)))
    rng = float(np.ceil(max_len + 0.5))
    xr, yr = [-rng, rng], [-rng, rng]

    fig = go.Figure()

    if mode.startswith("수동"):
        a = st.slider("a", -3.0, 3.0, 1.0, 0.1)
        b = st.slider("b", -3.0, 3.0, 1.0, 0.1)
        r = (a*v1 + b*v2).astype(float)

        # 초기 4 traces (v1, v2, result, locus)
        fig.add_trace(go.Scatter(x=[0, v1[0]], y=[0, v1[1]], mode="lines+markers", name="v1"))
        fig.add_trace(go.Scatter(x=[0, v2[0]], y=[0, v2[1]], mode="lines+markers", name="v2"))
        fig.add_trace(go.Scatter(x=[0, r[0]],  y=[0, r[1]],  mode="lines+markers", name="a·v1 + b·v2"))
        fig.add_trace(go.Scatter(x=[r[0]], y=[r[1]], mode="markers", name="locus", showlegend=False))

        # 평행사변형
        fig.add_trace(go.Scatter(
            x=[0, v1[0], r[0], v2[0], 0], y=[0, v1[1], r[1], v2[1], 0],
            fill="toself", mode="lines", name="parallelogram", showlegend=False, opacity=0.2
        ))

        fig.update_layout(title=f"a={a:.2f}, b={b:.2f}")

    else:
        # 애니메이션 안전 구성: 초기 4 traces + 각 frame에도 4개 항목 보장
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
            # numpy → list 변환으로 직렬화 안전성 확보
            locus_x = res[:i+1, 0].tolist()
            locus_y = res[:i+1, 1].tolist()
            frames.append(go.Frame(
                data=[
                    go.Scatter(),                              # v1 (변화 없음)
                    go.Scatter(),                              # v2 (변화 없음)
                    go.Scatter(x=[0, rx], y=[0, ry]),          # result
                    go.Scatter(x=locus_x, y=locus_y)           # locus
                ],
                name=f"t{i}",
                layout=go.Layout(title=f"a=cos t={a[i]:.2f},  b=sin t={b[i]:.2f}")
            ))
        fig.frames = frames

        fig.update_layout(
            updatemenus=[{
                "type": "buttons",
                "buttons": [
                    {"label": "▶ Play", "method": "animate",
                     "args": [None, {"frame": {"duration": int(speed), "redraw": True},
                                     "fromcurrent": True, "transition": {"duration": 0}}]},
                    {"label": "⏸ Pause", "method": "animate",
                     "args": [[None], {"mode": "immediate",
                                       "frame": {"duration": 0, "redraw": False},
                                       "transition": {"duration": 0}}]}
                ],
                "direction": "left", "x": 0.0, "y": 1.12
            }],
            sliders=[{
                "steps": [{"args": [[f"t{i}"],
                                     {"frame": {"duration": 0, "redraw": True},
                                      "mode": "immediate",
                                      "transition": {"duration": 0}}],
                           "label": f"{i}", "method": "animate"} for i in range(int(T))],
                "x": 0.05, "y": 1.04, "len": 0.9
            }]
        )

    # 공통 레이아웃
    fig.update_layout(
        xaxis=dict(range=xr, showgrid=show_grid, zeroline=True),
        yaxis=dict(range=yr, showgrid=show_grid, zeroline=True),
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(bgcolor="rgba(255,255,255,0.6)")
    )
    if keep_ratio:
        fig.update_yaxes(scaleanchor="x", scaleratio=1)

    st.plotly_chart(fig, use_container_width=True)


with tabs[8]:
    st.subheader("기초도구 (전기 · 분수)")
    tool = st.radio(
        "도구 선택",
        ["분수 더하기", "옴의 법칙(DC)", "AC 파형·위상(애니메이션)", "저항 직렬/병렬"],
        horizontal=True,
        key="basic_tool"
    )
    # 메뉴 전환 시 깨끗하게 초기화
    if st.session_state.get("current_tool") != tool:
        st.session_state["current_tool"] = tool
        st.session_state.pop("__modal_payload__", None)  # 대기 모달 삭제
        st.session_state.pop("pf_warned", None)          # 역률 경고 플래그 리셋

    # ---------- 1) 분수 더하기 ----------
    if tool == "분수 더하기":
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**분수 1**")
            n1 = st.number_input("분자₁", value=1, step=1, format="%d")
            d1 = st.number_input("분모₁(0 제외)", value=2, step=1, format="%d")
        with c2:
            st.markdown("**분수 2**")
            n2 = st.number_input("분자₂", value=1, step=1, format="%d")
            d2 = st.number_input("분모₂(0 제외)", value=3, step=1, format="%d")

        if d1 == 0 or d2 == 0:
            st.error("분모는 0이 될 수 없습니다.")
        else:
            L = _lcm(int(d1), int(d2))
            n_sum = int(n1)*(L//int(d1)) + int(n2)*(L//int(d2))
            nr, dr = simplify(n_sum, L)
            mix = to_mixed(nr, dr)

            st.latex(rf"""\frac{{{n1}}}{{{d1}}} + \frac{{{n2}}}{{{d2}}}
            = \frac{{{n1}\cdot{L//d1}}}{{{L}}} + \frac{{{n2}\cdot{L//d2}}}{{{L}}}
            = \frac{{{n_sum}}}{{{L}}}
            = \frac{{{nr}}}{{{dr}}}""")
            if mix:
                q, r, dd = mix
                st.markdown(f"**대답:** " + (f"{q}" if r==0 else f"대분수 **{q} {r}/{dd}** (기약분수 {nr}/{dr})"))

            st.divider()
            st.markdown("#### 🧩 연습 모드")
            if "frac_q" not in st.session_state:
                st.session_state.frac_q = (1, 2, 1, 3)
            if st.button("새 문제 뽑기"):
                import random
                st.session_state.frac_q = (
                    random.randint(-5,5) or 1, random.randint(1,9),
                    random.randint(-5,5) or 1, random.randint(1,9)
                )
                st.session_state.pop("frac_ok_shown", None)   # ← 정답 모달 표시 여부 리셋

            a1,b1,a2,b2 = st.session_state.frac_q
            st.write(f"문제: {a1}/{b1} + {a2}/{b2}")
            ua = st.text_input("정답(기약분수, 예: 5/6 또는 -7/3)", key="ua_input")

            ans_n, ans_d = add_fractions(a1,b1,a2,b2)
           

    # ---------- 2) 옴의 법칙(DC) ----------
    elif tool == "옴의 법칙(DC)":
        st.markdown("**V = I·R**,  **P = V·I**")
        col = st.columns(3)
        with col[0]: V = st.number_input("전압 V (Volt)", value=12.0, step=0.5)
        with col[1]: R = st.number_input("저항 R (Ohm)", value=6.0, step=0.5, min_value=0.0)
        with col[2]:
            mode_dc = st.selectbox("고정할 항목", ["V·R로 I 계산", "V·I로 R 계산", "I·R로 V 계산"])
        I = None
        if mode_dc == "V·R로 I 계산":
            V,I,R,P = ohm_dc_result(V=V, R=R, I=None)
        elif mode_dc == "V·I로 R 계산":
            I = st.number_input("전류 I (Ampere)", value=1.0, step=0.1)
            V,I,R,P = ohm_dc_result(V=V, I=I, R=None)
        else:
            I = st.number_input("전류 I (Ampere)", value=2.0, step=0.1)
            V,I,R,P = ohm_dc_result(V=None, I=I, R=R)
        st.info(f"**I = {I:.3f} A**,  **R = {R:.3f} Ω**,  **V = {V:.3f} V**,  **P = {P:.3f} W**")

    # ---------- 3) AC 파형·위상(애니메이션) ----------
    elif tool == "AC 파형·위상(애니메이션)":
        col = st.columns(4)
        with col[0]: Vrms = st.slider("전압 Vrms (V)", 1.0, 240.0, 220.0, 1.0)
        with col[1]: Irms = st.slider("전류 Irms (A)", 0.1, 20.0, 5.0, 0.1)
        with col[2]: f = st.slider("주파수 f (Hz)", 10.0, 120.0, 60.0, 1.0)
        with col[3]: load = st.selectbox("부하", ["저항성(R)", "유도성(L)", "용량성(C)", "사용자지정"])
        if load == "저항성(R)":   phi_deg = 0.0
        elif load == "유도성(L)": phi_deg = 90.0
        elif load == "용량성(C)": phi_deg = -90.0
        else:                     phi_deg = st.slider("위상차 φ (deg, V→I)", -180.0, 180.0, 30.0, 1.0)

        phi = math.radians(phi_deg)
        Vp = Vrms*math.sqrt(2); Ip = Irms*math.sqrt(2)
        PF = math.cos(phi); S = Vrms*Irms; P = S*PF; Q = S*math.sin(phi)
        st.caption(f"PF = cos φ = {PF:.3f},  유효전력 P = {P:.2f} W,  무효전력 Q = {Q:.2f} var,  피상전력 S = {S:.2f} VA")

        # 경고 모달 (히스테리시스 포함)
        if PF < 0.80 and not st.session_state.get("pf_warned", False):
            st.session_state["pf_warned"] = True
            trigger_modal({
                "title": "역률 경고 ⚡",
                "body": (
                    f"현재 역률 PF = **{PF:.2f}** (φ={phi_deg:.1f}°) 입니다.\n\n"
                    f"- 유효전력 P ≈ **{P:.1f} W**\n"
                    f"- 무효전력 Q ≈ **{Q:.1f} var**\n"
                    f"- 목표 PF 0.95 보상: Qc = P·(tanφ₁ − tanφ₂)"
                ),
                "key": "pf_warn"
            })
        elif PF >= 0.82 and st.session_state.get("pf_warned", False):
            st.session_state["pf_warned"] = False

        # 재생/정지
        if "ac_play" not in st.session_state: st.session_state.ac_play = False
        c1, c2 = st.columns([1,1])
        with c1:
            if st.button("▶ 재생"): st.session_state.ac_play = True
        with c2:
            if st.button("⏸ 정지"): st.session_state.ac_play = False

        left, right = st.columns(2)
        with left:  phL = st.empty()
        with right: phR = st.empty()

        if st.session_state.ac_play:
            secs = 3; fps = 30
            start = time.perf_counter()
            for k in range(secs*fps):
                if not st.session_state.ac_play: break
                phL.plotly_chart(phasor_fig(1.0, 1.0, phi, title=f"Phasor (φ={phi_deg:.1f}°)"), use_container_width=True)
                phR.plotly_chart(waveform_fig(Vp, Ip, f, phi, dur=2/f), use_container_width=True)
                sleep = (k+1)/fps - (time.perf_counter() - start)
                if sleep > 0: time.sleep(sleep)
            st.session_state.ac_play = False
        else:
            phL.plotly_chart(phasor_fig(1.0, 1.0, phi, title=f"Phasor (φ={phi_deg:.1f}°)"), use_container_width=True)
            phR.plotly_chart(waveform_fig(Vp, Ip, f, phi, dur=2/f), use_container_width=True)

    # ---------- 4) 저항 직렬/병렬 ----------
    elif tool == "저항 직렬/병렬":
        st.markdown("입력 예: `100, 220, 330` (Ω)")
        s = st.text_input("저항 값 목록 (콤마 구분)", "100, 220, 330")
        try:
            values = [float(x) for x in s.split(",") if x.strip()]
            rs, rp = series_parallel_req(values)
            st.info(f"**직렬 합성 Rₛ = {rs:.3f} Ω**,   **병렬 합성 Rₚ = {rp:.3f} Ω**")
            st.latex(r"R_{\text{series}} = \sum_i R_i \quad,\quad \frac{1}{R_{\text{parallel}}}=\sum_i \frac{1}{R_i}")
        except Exception:
            st.error("숫자만 콤마로 입력해주세요.")

st.markdown("---")
st.caption("이재오에게 저작권이 있으며 개발이나 협업하고자 하시는 관계자는 연락바랍니다")
# ── Global modal dispatcher (run once at end) ─────────────────────────────────
# ── Global modal dispatcher (run once at end) ─────────────────
payload = st.session_state.pop("__modal_payload__", None)  # ← get() 대신 pop()
if payload:
    open_modal(payload.get("title", "알림"),
               payload.get("body", ""),
               key=payload.get("key", "evt"))

