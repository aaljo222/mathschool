# tabs/tab_manim.py
import streamlit as st
from pathlib import Path

# 배포환경에서 manim이 없을 수도 있으므로 try/except
try:
    from manim import Scene, Square, Circle, BLUE, WHITE, Create, Transform
    from utils.manim_runner import render_manim
    _HAS_MANIM = True
except Exception as e:
    _HAS_MANIM = False
    _ERR = e

FALLBACK_MP4 = Path("public/videos/square_to_circle.mp4")  # 미리 렌더해 둔 파일(선택)

class SquareToCircle(Scene):
    def construct(self):
        s = Square().set_fill(BLUE, 0.5).set_stroke(WHITE, 2)
        self.play(Create(s))
        self.play(Transform(s, Circle()))
        self.wait(0.2)

def render():
    st.subheader("Manim 데모 (경량 렌더)")

    col1, col2, col3 = st.columns(3)
    with col1: w   = st.number_input("폭(px)", 320, 1280, 640, 10)
    with col2: h   = st.number_input("높이(px)", 180, 720, 360, 10)
    with col3: fps = st.slider("FPS", 5, 30, 20)

    # 1회 렌더 방지(세션 중 이미 렌더했다면 재활용)
    key = "square_to_circle_last"
    if st.button("🎬 렌더 실행"):
        if not _HAS_MANIM:
            st.error(f"이 환경에서는 manim을 사용할 수 없습니다: {_ERR}")
        else:
            with st.spinner("Manim 렌더링 중..."):
                try:
                    path = render_manim(SquareToCircle, cache_key="square_to_circle",
                                        w=int(w), h=int(h), fps=int(fps))
                    st.session_state[key] = str(path)
                except Exception as e:
                    st.error(f"렌더 실패: {e}")

    # 우선순위: 세션에 방금 렌더한 최신 경로 → fallback mp4 → 안내
    path_str = st.session_state.get(key)
    if path_str and Path(path_str).exists():
        st.video(path_str)
    elif FALLBACK_MP4.exists():
        st.info("로컬에 미리 렌더된 예제 영상을 표시합니다.")
        st.video(str(FALLBACK_MP4))
    else:
        st.warning("아직 렌더된 영상이 없습니다. '🎬 렌더 실행'을 눌러주세요.")
