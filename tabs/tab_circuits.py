# tabs/tab_circuits.py
from pathlib import Path
import streamlit as st
from utils.media import show_gif_cached, WIX_HEADERS  # 재사용!

# 보여줄 GIF들 (제목, URL, 로컬 파일명, 헤더)
GIFS = [
    ("스위치 ON/OFF",
     "https://4.bp.blogspot.com/-D3B-1rJwkkw/W90uWN__3wI/AAAAAAAAVHs/MpSDNjDDKbUggOvdExg70eSjAdM7p3voACLcBGAs/s640/switch%2BON%2BOFF.gif",
     "switch_on_off.gif", None),

    ("nMOS 스위치",
     "https://file.notion.so/f/f/951cbd31-d9bf-4a9b-b47e-9583f9f44324/36b5910c-61d4-4c23-974c-45a5e84b039e/nMos_sw_gif_(1).gif?table=block&id=1f1aff74-c7ba-8050-8891-d900fee9581b&spaceId=951cbd31-d9bf-4a9b-b47e-9583f9f44324&expirationTimestamp=1758960000000&signature=hAWaUiOuWMi1RkAwYew1-zIJHaocWACffSa-9vG23h4",
     "nmos_switch.gif", None),

    ("트랜지스터 스위치 동작",
     "https://file.notion.so/f/f/951cbd31-d9bf-4a9b-b47e-9583f9f44324/2f82a011-b495-479a-8d15-dde14c160b3e/Transistor-as-switch-working..gif.b754ae14ba7257b854db964f1c1b9db1.gif?table=block&id=1f1aff74-c7ba-80e9-8348-d68c17cfeaeb&spaceId=951cbd31-d9bf-4a9b-b47e-9583f9f44324&expirationTimestamp=1758960000000&signature=b7bUaVGjvTg5YPq1KBu0FGmtrCW8wo9GSxQPqkTnWq4",
     "bjt_switch.gif", None),

    ("트랜지스터 기초",
     "https://file.notion.so/f/f/951cbd31-d9bf-4a9b-b47e-9583f9f44324/0884f12e-63b9-4701-978f-8553257451d8/transistr1.gif?table=block&id=1f1aff74-c7ba-80a8-bb3b-ecda9ea17b1a&spaceId=951cbd31-d9bf-4a9b-b47e-9583f9f44324&expirationTimestamp=1758960000000&signature=dljsiBUQWND2Um-rafnL5itLlrw_oMebF7vpNbzDZu0",
     "transistor_basic.gif", None),

    ("푸시풀 증폭기",
     "https://file.notion.so/f/f/951cbd31-d9bf-4a9b-b47e-9583f9f44324/51459c97-1e6d-4d03-9060-194d12eb6eb9/push-pull-amplifier.gif?table=block&id=1f1aff74-c7ba-8058-9a05-f3f72994a334&spaceId=951cbd31-d9bf-4a9b-b47e-9583f9f44324&expirationTimestamp=1758960000000&signature=t1CO7wzpNoEkiqbyLn8fUQ6_rJQfLMBqzjfWQWrQeEE",
     "push_pull.gif", None),

    ("로봇맨 애니메(모터/기구)",
     "https://file.notion.so/f/f/951cbd31-d9bf-4a9b-b47e-9583f9f44324/60eda1ba-80aa-424e-aa3e-00c2349b81a6/Fig2_-_RobotManAnim.gif?table=block&id=1f1aff74-c7ba-8006-a99f-e63b617fb7ec&spaceId=951cbd31-d9bf-4a9b-b47e-9583f9f44324&expirationTimestamp=1758960000000&signature=zURJhbjDoFHprVMD1Y0aOGyQz1idpSaAlnF0oJX62nU",
     "robot_anim.gif", None),
]

def render():
    st.subheader("회로 애니메이션 모음")
    cols = st.columns(2)
    for i, (title, url, fname, headers) in enumerate(GIFS):
        with cols[i % 2]:
            show_gif_cached(
                url,
                filename=fname,
                caption=title,
                headers=headers,          # 필요하면 WIX_HEADERS 등 전달
                subdir="circuits",        # 캐시 하위 폴더
                show_link_on_fail=True,   # 실패 시 URL 링크 안내
            )
