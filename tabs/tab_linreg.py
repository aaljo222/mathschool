# tabs/tab_linreg.py
import numpy as np
import streamlit as st
import plotly.graph_objects as go

def render():
    st.subheader("선형회귀: y = a x + b (경사하강 · 이벤트 1회 갱신)")

    c1, c2, c3 = st.columns(3)
    with c1:
        true_a = st.slider("진짜 기울기 (데이터 생성)", -5.0, 5.0, 2.0, 0.1)
        true_b = st.slider("진짜 절편 (데이터 생성)", -10.0, 10.0, -1.0, 0.1)
        seed   = st.slider("시드", 0, 9999, 0, 1)
    with c2:
        noise  = st.slider("노이즈 표준편차", 0.0, 5.0, 1.0, 0.1)
        npts   = st.slider("데이터 개수", 10, 400, 120, 10)
        x_span = st.slider("x 범위(±)", 1.0, 10.0, 5.0, 0.5)
    with c3:
        lr     = st.slider("학습률(learning rate)", 1e-5, 0.5, 0.05, 0.0005, format="%.5f")
        steps  = st.slider("경사하강 스텝 수", 1, 2000, 300, 10)
        k_show = st.slider("표시 스텝 k", 0, steps, steps, 1)

    opt1, opt2 = st.columns(2)
    with opt1:
        show_ols = st.checkbox("정규방정식(해석해) 라인 표시", True)
    with opt2:
        show_land = st.checkbox("손실 지형(Contour) + (a,b) 경로", False,
                                help="켜면 (a,b) 공간의 MSE 등고선과 경사하강 경로를 봅니다.")

    # ----- 데이터 생성 -----
    rng = np.random.default_rng(seed)
    x = np.linspace(-x_span, x_span, npts)
    y = true_a * x + true_b + rng.normal(0, noise, size=npts)

    # ----- 손실/그라디언트 -----
    def mse(a, b):
        return float(np.mean((y - (a*x + b))**2))
    def grad(a, b):
        n = len(x)
        e = (a*x + b) - y
        return float((e @ x) / n), float(np.sum(e) / n)  # dL/da, dL/db

    # ----- 경사하강 경로(한 번에 전부 계산해두고 k로 보여줌) -----
    a_hist = np.empty(steps+1); b_hist = np.empty(steps+1); m_hist = np.empty(steps+1)
    a_hist[0], b_hist[0] = 0.0, 0.0
    m_hist[0] = mse(a_hist[0], b_hist[0])
    for t in range(1, steps+1):
        da, db = grad(a_hist[t-1], b_hist[t-1])
        a_hist[t] = a_hist[t-1] - lr*da
        b_hist[t] = b_hist[t-1] - lr*db
        m_hist[t] = mse(a_hist[t], b_hist[t])

    a_k, b_k = a_hist[k_show], b_hist[k_show]

    # ----- OLS(해석해) -----
    if show_ols:
        X1 = np.vstack([x, np.ones_like(x)]).T
        a_ols, b_ols = np.linalg.lstsq(X1, y, rcond=None)[0]
        mse_ols = mse(a_ols, b_ols)
    else:
        a_ols = b_ols = mse_ols = None

    # ----- (1) 데이터 + 직선들 -----
    fig_lr = go.Figure()
    fig_lr.add_trace(go.Scatter(x=x, y=y, mode="markers", name="data",
                                marker=dict(size=7, opacity=0.7)))
    fig_lr.add_trace(go.Scatter(x=x, y=true_a*x+true_b, mode="lines", name="true line"))
    fig_lr.add_trace(go.Scatter(x=x, y=a_k*x+b_k, mode="lines", name=f"GD step {k_show}",
                                line=dict(width=3)))
    if show_ols:
        fig_lr.add_trace(go.Scatter(x=x, y=a_ols*x+b_ols, mode="lines",
                                    name="OLS(해석해)", line=dict(dash="dot")))
    ttl = f"GD 결과: a≈{a_k:.3f}, b≈{b_k:.3f},  MSE≈{m_hist[k_show]:.4f}"
    if show_ols:
        ttl += f"  |  OLS: a≈{a_ols:.3f}, b≈{b_ols:.3f}, MSE≈{mse_ols:.4f}"
    fig_lr.update_layout(title=ttl, template="plotly_white", height=480)

    # ----- (2) MSE 히스토리 -----
    fig_mse = go.Figure()
    fig_mse.add_trace(go.Scatter(x=np.arange(0, steps+1), y=m_hist, mode="lines", name="MSE"))
    fig_mse.add_vline(x=k_show, line_dash="dot")
    fig_mse.update_layout(title="MSE 감소(경사하강 경로)", xaxis_title="step", yaxis_title="MSE",
                          template="plotly_white", height=320)

    cplot1, cplot2 = st.columns(2)
    with cplot1: st.plotly_chart(fig_lr, use_container_width=True)
    with cplot2: st.plotly_chart(fig_mse, use_container_width=True)

    # ----- (3) 손실 지형(선택) -----
    if show_land:
        # a,b 범위를 데이터/진실 주변에서 자동 추정
        a_min, a_max = true_a - 4, true_a + 4
        b_min, b_max = true_b - 6, true_b + 6
        aa = np.linspace(a_min, a_max, 80)
        bb = np.linspace(b_min, b_max, 80)
        A, B = np.meshgrid(aa, bb)
        # MSE(a,b) = mean((y - (a x + b))^2)
        Yhat = A[..., None]*x + B[..., None]
        Z = np.mean((y - Yhat)**2, axis=2)

        fig_land = go.Figure(
            data=go.Contour(x=aa, y=bb, z=Z, colorscale="Viridis",
                            contours=dict(showlabels=True))
        )
        # 경사하강 경로
        fig_land.add_trace(go.Scatter(x=a_hist, y=b_hist, mode="lines+markers",
                                      name="GD path", line=dict(width=3), marker=dict(size=5)))
        # 현재점
        fig_land.add_trace(go.Scatter(x=[a_k], y=[b_k], mode="markers",
                                      name="now", marker=dict(size=10, symbol="diamond")))
        # OLS 점
        if show_ols:
            fig_land.add_trace(go.Scatter(x=[a_ols], y=[b_ols], mode="markers",
                                          name="OLS", marker=dict(size=10, symbol="x")))
        fig_land.update_layout(
            template="plotly_white", height=520, title="손실 지형 MSE(a,b)와 경사하강 경로",
            xaxis_title="a", yaxis_title="b"
        )
        st.plotly_chart(fig_land, use_container_width=True)
