import math, time
import streamlit as st
from utils.frac import add_fractions, simplify, to_mixed, _lcm
from utils.elec import ohm_dc_result, series_parallel_req
from utils.plot import phasor_fig, waveform_fig, make_parallel_animation


# --- 탭 렌더러 ---
def render():

    st.subheader("기초도구 (전기 · 분수)")
    tool = st.radio(
        "도구 선택",
        ["분수 더하기", "옴의 법칙(DC)", "AC 파형·위상(애니메이션)", "저항 직렬/병렬", "병렬 저항(애니메이션)"],
        horizontal=True,
        key="basic_tool",
    )

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
                q, rmd, dd = mix
                st.markdown("**대답:** " + (f"{q}" if rmd==0 else f"대분수 **{q} {rmd}/{dd}** (기약분수 {nr}/{dr})"))

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

            a1, b1, a2, b2 = st.session_state.frac_q
            st.write(f"문제: {a1}/{b1} + {a2}/{b2}")
            ua = st.text_input("정답(기약분수, 예: 5/6 또는 -7/3)", key="ua_input")

            ans_n, ans_d = add_fractions(a1, b1, a2, b2)
            if ua.strip():
                try:
                    sn, sd = map(int, ua.replace(" ","").split("/"))
                    sn, sd = simplify(sn, sd)
                    if (sn, sd) == (ans_n, ans_d):
                        st.success("정답! ✅")
                    else:
                        st.error(f"오답 ❌  정답: {ans_n}/{ans_d}")
                except Exception:
                    st.warning(f"형식이 올바르지 않습니다. 정답: {ans_n}/{ans_d}")

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
            V, I, R, P = ohm_dc_result(V=V, R=R, I=None)
        elif mode_dc == "V·I로 R 계산":
            I = st.number_input("전류 I (Ampere)", value=1.0, step=0.1)
            V, I, R, P = ohm_dc_result(V=V, I=I, R=None)
        else:
            I = st.number_input("전류 I (Ampere)", value=2.0, step=0.1)
            V, I, R, P = ohm_dc_result(V=None, I=I, R=R)

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

        if PF < 0.80:
            st.warning("역률 PF가 0.80 미만입니다. 콘덴서 보상(Qc = P·(tanφ₁ − tanφ₂))을 검토하세요.")

        if "ac_play" not in st.session_state:
            st.session_state.ac_play = False
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
                phL.plotly_chart(phasor_fig(1.0, 1.0, phi, title=f"Phasor (φ={phi_deg:.1f}°)"),
                                 use_container_width=True)
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

    # ---------- 5) 병렬 저항(애니메이션) ----------
    elif tool == "병렬 저항(애니메이션)":
        st.markdown("DC 병렬 회로에서 **모든 가지의 전압은 동일**하고, 전류는 **각 저항에 반비례**합니다.")
        col = st.columns([1, 1, 1])
        with col[0]:
            V = st.number_input("공급 전압 V (Volt)", value=12.0, step=0.5, min_value=0.0)
        with col[1]:
            s_R = st.text_input("저항 목록 R (Ω, 콤마로)", "100, 220, 330")
        with col[2]:
            secs = st.slider("길이(초)", 2, 10, 5)
            fps  = st.slider("FPS", 10, 40, 20)

        try:
            Rs = [float(x) for x in s_R.split(",") if x.strip()]
            if len(Rs) == 0:
                st.warning("저항을 1개 이상 입력하세요.")
                return

            I_each = [V / r for r in Rs]
            I_tot  = sum(I_each)
            Req    = 1.0 / sum(1.0 / r for r in Rs)

            rows = [
                {"가지": f"{i+1}", "R (Ω)": f"{Rs[i]:.3f}", "I (A)": f"{I_each[i]:.3f}",
                 "P = V·I (W)": f"{(V*I_each[i]):.3f}"}
                for i in range(len(Rs))
            ]
            st.table(rows)
            st.info(f"합성저항 **Req = {Req:.3f} Ω**,  전체전류 **I_total = {I_tot:.3f} A** (검산: I_total = V/Req)")

            fig = make_parallel_animation(Rs, V, seconds=secs, fps=fps)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(r"""
**계산 과정**
1) 병렬에서 가지 전압은 동일:  \(V_1 = V_2 = \cdots = V\).  
2) 가지 전류:  \( I_i = \dfrac{V}{R_i} \).  
3) 전체 전류(KCL):  \( I_{\text{total}} = \sum_i I_i \).  
4) 합성저항:  \( \dfrac{1}{R_{\mathrm{eq}}} = \sum_i \dfrac{1}{R_i} \Rightarrow R_{\mathrm{eq}} = \dfrac{1}{\sum_i 1/R_i} \).  
5) 검산:  \( I_{\text{total}} = \dfrac{V}{R_{\mathrm{eq}}} \).
""")
        except Exception:
            st.error("입력 형식을 확인하세요. 예: 100, 220, 330")
