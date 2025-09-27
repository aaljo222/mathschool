import streamlit as st
from utils import style

# tabs 모듈들
from tabs import (
    tab_conic, tab_trig, tab_calc_def, tab_linreg, tab_taylor, tab_fft, tab_euler,
    tab_exp_log,
    tab_basics,        # ← 여기 추가 (파일: tabs/tab_basic.py)
    tab_lincomb3,     # ← 여기 추가 (파일: tabs/tab_lincomb3.py)
    tab_circuits
)
# utils에 남아있는 탭
from utils import tab_vectors as tab_vec_combo

# (선택) manim 탭 존재 시만 로드
_HAS_MANIM = False
try:
    from tabs import tab_manim
    _HAS_MANIM = True
except Exception:
    _HAS_MANIM = False

st.set_page_config(page_title="수학 애니메이션 튜터", layout="wide")
style.inject_css()
style.show_notice_banner()

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
    "3D 열벡터 합(연립방정식)",
    "회로",  # ← 추가
]

TAB_MODULES = [
    tab_conic, tab_trig, tab_calc_def, tab_linreg, tab_taylor, tab_fft, tab_euler,
    tab_exp_log,
    tab_vec_combo,     # utils.tab_vectors
    tab_basics,         # tabs/tab_basic.py  (이전의 tab_basics 아님)
    tab_lincomb3,      # tabs/tab_lincomb3.py
    tab_circuits
]

if _HAS_MANIM:
    TAB_TITLES.append("Manim 데모")
    TAB_MODULES.append(tab_manim)

left, right = st.columns([1, 3])
with left:
    nav_mode = st.radio("메뉴 보기", ["탭", "콤팩트"], index=0, horizontal=True, key="nav_mode")
with right:
    sel = None
    if nav_mode == "콤팩트":
        default_idx = st.session_state.get("nav_idx", 0)
        sel = st.selectbox("이동할 메뉴", TAB_TITLES, index=default_idx, key="nav_select")

def _render_by_index(idx: int):
    TAB_MODULES[idx].render()

if nav_mode == "탭":
    tabs = st.tabs(TAB_TITLES)
    for i, t in enumerate(tabs):
        with t:
            _render_by_index(i)
else:
    idx = TAB_TITLES.index(sel or TAB_TITLES[0])
    st.session_state["nav_idx"] = idx
    _render_by_index(idx)

st.markdown("---")
st.caption("이재오에게 저작권이 있으며 개발이나 협업하고자 하시는 관계자는 연락바랍니다")
