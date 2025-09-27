# utils/anim.py
import time
import streamlit as st

def next_frame_index(pfx: str, steps: int, fps: int, autorun: bool) -> int:
    """Play 버튼 없이 자동재생/정지.
    - 실행당 프레임 하나만 그린 뒤 autorun이면 k를 1 증가시키고 rerun.
    - pfx는 탭별 고유 접두사(세션키 충돌 방지).
    """
    k_key = f"{pfx}:k"
    if k_key not in st.session_state:
        st.session_state[k_key] = 0

    k = st.session_state[k_key] % max(1, steps)

    if autorun:
        st.session_state[k_key] = (k + 1) % max(1, steps)
        time.sleep(1.0 / max(1, fps))
        try:
            st.rerun()                # 최신 버전
        except Exception:
            st.experimental_rerun()   # 구버전 호환

    return k
