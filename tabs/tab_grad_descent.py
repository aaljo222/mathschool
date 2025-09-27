# tabs/tab_grad_descent.py
import numpy as np, streamlit as st, plotly.graph_objects as go

PFX = "gd2"

def render():
    st.subheader("경사하강법 (2D 쿼드라틱)")

    # f(x,y) = 1/2 (x,y)^T A (x,y)
    a11 = st.slider("a11", 0.2, 4.0, 2.0, 0.1, key=f"{PFX}:a11")
    a22 = st.slider("a22", 0.2, 4.0, 0.8, 0.1, key=f"{PFX}:a22")
    a12 = st.slider("a12", -1.0, 1.0, 0.3, 0.05, key=f"{PFX}:a12")
    A = np.array([[a11, a12], [a12, a22]])

    x0 = np.array([
        st.slider("x₀", -3.0, 3.0,  2.0, 0.1, key=f"{PFX}:x0"),
        st.slider("y₀", -3.0, 3.0, -1.5, 0.1, key=f"{PFX}:y0"),
    ])
    lr    = st.slider("학습률 η", 0.01, 0.5, 0.12, 0.01, key=f"{PFX}:lr")
    steps = st.slider("스텝 수", 5, 120, 60, key=f"{PFX}:steps")
    prog  = st.slider("진행도", 0.0, 1.0, 1.0, 0.01, key=f"{PFX}:prog",
                      help="0이면 시작점, 1이면 steps까지의 경로 전체를 표시")

    # 수렴 가이드 (λ_max로 0<η<2/λ_max 권장)
    w, _ = np.linalg.eigh(A)               # A는 대칭
    lam_max = float(np.max(w))
    eta_crit = 2.0/lam_max if lam_max > 1e-12 else np.inf
    st.caption(f"λ_max ≈ {lam_max:.3f} → 권장 범위: 0 < η < 2/λ_max ≈ {eta_crit:.3f}")

    # 경로 미리 계산 (한 번)
    xs = [x0]
    for _ in range(steps-1):
        g = A @ xs[-1]
        xs.append(xs[-1] - lr * g)
    xs = np.array(xs)

    # 진행도에 해당하는 스텝만 보여주기
    k = int(round(prog * (steps - 1)))

    # 표시 범위 자동 조절
    max_abs = float(np.max(np.abs(xs))) if xs.size else 3.0
    R = max(3.0, min(10.0, 1.5 * max_abs))

    # 등고선
    X = np.linspace(-R, R, 160)
    Y = np.linspace(-R, R, 160)
    XX, YY = np.meshgrid(X, Y)
    ZZ = 0.5*(A[0,0]*XX**2 + 2*A[0,1]*XX*YY + A[1,1]*YY**2)

    fig = go.Figure(data=go.Contour(
        x=X, y=Y, z=ZZ, contours_coloring="lines", showscale=False
    ))
    fig.add_scatter(
        x=xs[:k+1,0], y=xs[:k+1,1],
        mode="lines+markers", name="path", line=dict(width=3)
    )
    fig.update_layout(
        template="plotly_white", height=520,
        xaxis=dict(range=[-R, R]),
        yaxis=dict(range=[-R, R], scaleanchor="x", scaleratio=1),
        title=f"step {k}/{steps-1} | f(x)=½ xᵀAx"
    )

    st.plotly_chart(fig, use_container_width=True)

    fk = 0.5 * xs[k] @ (A @ xs[k])
    st.caption(f"x_k ≈ ({xs[k,0]:.3f}, {xs[k,1]:.3f}),   f(x_k) ≈ {fk:.5f}")
    if lr >= eta_crit and np.isfinite(eta_crit):
        st.warning(f"η={lr:.3f} ≥ 2/λ_max≈{eta_crit:.3f}이면 발산할 수 있어요.")
