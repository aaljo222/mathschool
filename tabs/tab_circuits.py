# tabs/tab_circuits.py
from pathlib import Path
import streamlit as st
from utils.media import cache_remote_asset, show_image

# 제목, 원본 URL, 로컬 파일명
CIRCUIT_GIFS = [
    ("스위치 ON/OFF",
     "https://4.bp.blogspot.com/-D3B-1rJwkkw/W90uWN__3wI/AAAAAAAAVHs/MpSDNjDDKbUggOvdExg70eSjAdM7p3voACLcBGAs/s640/switch%2BON%2BOFF.gif",
     "switch_on_off.gif"),
    ("nMOS 스위치",
     "https://file.notion.so/f/f/951cbd31-d9bf-4a9b-b47e-9583f9f44324/36b5910c-61d4-4c23-974c-45a5e84b039e/nMos_sw_gif_(1).gif?table=block&id=1f1aff74-c7ba-8050-8891-d900fee9581b&spaceId=951cbd31-d9bf-4a9b-b47e-9583f9f44324&expirationTimestamp=1758960000000&signature=hAWaUiOuWMi1RkAwYew1-zIJHaocWACffSa-9vG23h4",
     "nmos_switch.gif"),
    ("트랜지스터 스위치 동작",
     "https://file.notion.so/f/f/951cbd31-d9bf-4a9b-b47e-9583f9f44324/2f82a011-b495-479a-8d15-dde14c160b3e/Transistor-as-switch-working..gif.b754ae14ba7257b854db964f1c1b9db1.gif?table=block&id=1f1aff74-c7ba-80e9-8348-d68c17cfeaeb&spaceId=951cbd31-d9bf-4a9b-b47e-9583f9f44324&expirationTimestamp=1758960000000&signature=b7bUaVGjvTg5YPq1KBu0FGmtrCW8wo9GSxQPqkTnWq4",
     "bjt_switch.gif"),
    ("트랜지스터 동작(기본)",
     "https://file.notion.so/f/f/951cbd31-d9bf-4a9b-b47e-9583f9f44324/0884f12e-63b9-4701-978f-8553257451d8/transistr1.gif?table=block&id=1f1aff74-c7ba-80a8-bb3b-ecda9ea17b1a&spaceId=951cbd31-d9bf-4a9b-b47e-9583f9f44324&expirationTimestamp=1758960000000&signature=dljsiBUQWND2Um-rafnL5itLlrw_oMebF7vpNbzDZu0",
     "transistor_basic.gif"),
    ("푸시풀 증폭기",
     "https://file.notion.so/f/f/951cbd31-d9bf-4a9b-b47e-9583f9f44324/51459c97-1e6d-4d03-9060-194d12eb6eb9/push-pull-amplifier.gif?table=block&id=1f1aff74-c7ba-8058-9a05-f3f72994a334&spaceId=951cbd31-d9bf-4a9b-b47e-9583f9f44324&expirationTimestamp=1758960000000&signature=t1CO7wzpNoEkiqbyLn8fUQ6_rJQfLMBqzjfWQWrQeEE",
     "push_pull.gif"),
    ("로봇 팔 예시",
     "https://file.notion.so/f/f/951cbd31-d9bf-4a9b-b47e-9583f9f44324/60eda1ba-80aa-424e-aa3e-00c2349b81a6/Fig2_-_RobotManAnim.gif?table=block&id=1f1aff74-c7ba-8006-a99f-e63b617fb7ec&spaceId=951cbd31-d9bf-4a9b-b47e-9583f9f44324&expirationTimestamp=1758960000000&signature=zURJhbjDoFHprVMD1Y0aOGyQz1idpSaAlnF0oJX62nU",
     "robot_arm.gif"),
]

SUBDIR = "circuits"  # 캐시 하위폴더

def _display_one(title: str, url: str, fname: str):
    try:
        p = cache_remote_asset(url, fname, subdir=SUBDIR)
        show_image(str(p), caption=title)
    except Exception as e:
        st.warning(f"로컬 캐시 실패: {e}\n직접 URL로 표시합니다.")
        show_image(url, caption=title)

def render():
    st.subheader("회로 애니메이션 모음")

    mode = st.radio("보기 모드", ["한 장 크게", "모두 보기(썸네일)"], horizontal=True, key="circuit_mode")

    if mode == "한 장 크게":
        titles = [t for t,_,_ in CIRCUIT_GIFS]
        idx = st.selectbox("선택", titles, index=0)
        title, url, fname = CIRCUIT_GIFS[titles.index(idx)]
        _display_one(title, url, fname)

    else:
        # 3열 그리드로 썸네일 표시
        cols = st.columns(3)
        for i, (title, url, fname) in enumerate(CIRCUIT_GIFS):
            with cols[i % 3]:
                _display_one(title, url, fname)
