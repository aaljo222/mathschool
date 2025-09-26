# utils/style.py
import streamlit as st

def inject_css():
    st.markdown("""
    <style>
    .stTabs [role="tablist"]{gap:.25rem;overflow-x:auto;padding:.25rem 0;scrollbar-width:thin;flex-wrap:wrap;}
    .stTabs [role="tab"]{flex:0 0 auto;font-size:.95rem;padding:.35rem .7rem;}
    .notice{
      background:#fff8e6;border:1px solid #ffd7a1;border-left:8px solid #ff8b00;
      padding:12px 16px;border-radius:10px;margin:10px 0 18px 0;
    }
    .notice h3{margin:0 0 6px 0}
    </style>
    """, unsafe_allow_html=True)

def show_notice_banner():
    st.markdown(
        """
        <div class="notice">
          <h3>📢 교육 콘텐츠 개발 안내</h3>
          <div style="font-size:0.95rem; line-height:1.55">
            • 이 앱은 <b>매주 새로운 수학 애니메이션</b>을 추가합니다.<br/>
            • 중·고등 수학과 <b>전기 기능사</b> 학습 보조 도구를 제공합니다.<br/>
            • 맞춤형 <b>교육 콘텐츠 개발</b>·커리큘럼 제작 문의:
            <a href="mailto:aaljo2@naver.com"><b>aaljo2@naver.com</b></a>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )
