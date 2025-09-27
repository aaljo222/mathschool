# utils/anim.py
import time
import streamlit as st

def playbar(key: str = "anim", play_label: str = "▶ Play", stop_label: str = "⏸ Pause") -> bool:
    """재생/정지 토글 UI. 반환값이 True면 '재생 중'."""
    cols = st.columns([1, 1, 8])
    if key not in st.session_state:
        st.session_state[key] = False
    if cols[0].button(play_label, key=f"{key}_play"):
        st.session_state[key] = True
    if cols[1].button(stop_label, key=f"{key}_stop"):
        st.session_state[key] = False
    return st.session_state[key]

def step_loop(n_frames: int, fps: int = 24, key: str = "anim"):
    """재생 중일 때 프레임 인덱스를 yield. 중간 정지 시 루프 즉시 종료."""
    start = time.perf_counter()
    for i in range(n_frames):
        if not st.session_state.get(key, False):
            break
        yield i
        sleep = (i + 1) / fps - (time.perf_counter() - start)
        if sleep > 0:
            time.sleep(sleep)
    st.session_state[key] = False
