# app.py
import streamlit as st

# ── 공통 스타일/배너 ─────────────────────────────────────────────────────────
from utils import style  # style.inject_css(), style.show_notice_banner()

# ── 탭 모듈들 (tabs 폴더) ───────────────────────────────────────────────────
from tabs import (
    tab_conic, tab_trig, tab_calc_def, tab_linreg, tab_taylor, tab_fft, tab_euler,
    tab_exp_log,                      # ← 추가
)
# 기존 utils에 남아있는 탭
from utils import tab_vectors as tab_vec_combo   # 벡터 선형결합
from tabs import tab_lincomb3
# 선택: Manim 탭 존재 시만 노출 (없으면 자동 건너뜀)
_HAS_MANIM = False
try:
    from tabs import tab_manim  # 새로 만든 Manim 탭(있을 때만)
    _HAS_MANIM = True
except Exception:
    _HAS_MANIM = False

# ── 페이지 설정 & 상단 공지 ────────────────────────────────────────────────
st.set_page_config(page_title="수학 애니메이션 튜터", layout="wide")
style.inject_css()             # 탭 가로 스크롤/2줄 랩 등 공통 CSS
style.show_notice_banner()     # 상단 고정 공지

# ── 탭 타이틀/모듈 매핑 ────────────────────────────────────────────────────
TAB_TITLES = [
    "포물선/쌍곡선",
    "삼각함수",
    "미분·적분(정의)",
    "선형회귀",
    "테일러 시리즈",
    "푸리에 변환",
    "오일러 공식(애니메이션)",
    "지수·로그(쌍대)",
    "벡터의 선형결합",
    "기초도구(전기·분수)",
    "3D 열벡터 합(연립방정식)"
]

TAB_MODULES = [
    tab_conic, tab_trig, tab_calc_def, tab_linreg, tab_taylor, tab_fft, tab_euler,
    tab_exp_log,                      # ← 추가
    tab_vec_combo, tab_basics,
]

# Manim 탭이 준비된 경우 리스트에 추가
if _HAS_MANIM:
    TAB_TITLES.append("Manim 데모")
    TAB_MODULES.append(tab_manim)

# ── 네비게이션 모드(탭 / 콤팩트) ───────────────────────────────────────────
left, right = st.columns([1, 3])
with left:
    nav_mode = st.radio(
        "메뉴 보기",
        ["탭", "콤팩트"],
        index=0,
        horizontal=True,
        key="nav_mode",
        help="탭이 너무 많아질 때는 '콤팩트' 모드를 사용하세요.",
    )
with right:
    sel = None
    if nav_mode == "콤팩트":
        # 마지막 선택 기억
        default_idx = st.session_state.get("nav_idx", 0)
        sel = st.selectbox("이동할 메뉴", TAB_TITLES, index=default_idx, key="nav_select")

# ── 렌더러 ─────────────────────────────────────────────────────────────────
def _render_by_index(idx: int):
    TAB_MODULES[idx].render()

# ── 본문: 탭 혹은 콤팩트 렌더 ─────────────────────────────────────────────
if nav_mode == "탭":
    tabs = st.tabs(TAB_TITLES)
    for i, t in enumerate(tabs):
        with t:
            _render_by_index(i)
else:
    idx = TAB_TITLES.index(sel or TAB_TITLES[0])
    st.session_state["nav_idx"] = idx
    _render_by_index(idx)

# ── 푸터 ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("이재오에게 저작권이 있으며 개발이나 협업하고자 하시는 관계자는 연락바랍니다")
