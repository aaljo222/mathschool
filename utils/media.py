# utils/media.py
from __future__ import annotations
from pathlib import Path
import hashlib
import requests
import streamlit as st

# 캐시 저장소(프로젝트 안 정적 폴더)
ASSET_ROOT = Path("public/assets")
ASSET_ROOT.mkdir(parents=True, exist_ok=True)

# 일부 CDN(예: wixmp/DeviantArt)은 리퍼러 없으면 401/403을 반환
DEFAULT_HEADERS = {"User-Agent": "Mozilla/5.0"}
WIX_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://www.deviantart.com/",
}

def _derive_filename(url: str) -> str:
    """URL에서 파일명 추출 실패 시 해시로 이름 생성"""
    base = url.split("?")[0].rstrip("/").split("/")[-1]
    if base and "." in base:
        return base
    return hashlib.md5(url.encode()).hexdigest() + ".bin"

def cache_remote(
    url: str,
    filename: str | None = None,
    headers: dict | None = None,
    subdir: str = "",
    timeout=(5, 20),
) -> Path:
    """원격 파일을 로컬 캐시에 저장하고 Path 반환"""
    fname = filename or _derive_filename(url)
    out = ASSET_ROOT.joinpath(subdir, fname)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        r = requests.get(url, headers=headers or DEFAULT_HEADERS, timeout=timeout)
        # 핫링크 차단을 명확하게 알려줌
        if r.status_code in (401, 403):
            raise RuntimeError(f"blocked_by_host({r.status_code}) for url={url}")
        r.raise_for_status()
        out.write_bytes(r.content)
    return out

def show_image(src: str | Path, caption: str | None = None, clamp: bool = False):
    """Streamlit 버전 차이 대응 (폭 옵션 자동 선택)"""
    try:
        st.image(src, caption=caption, use_container_width=True, clamp=clamp)
    except TypeError:
        st.image(src, caption=caption, use_column_width=True, clamp=clamp)

def show_gif_cached(
    url: str,
    filename: str | None = None,
    caption: str | None = None,
    headers: dict | None = None,
    subdir: str = "",
    show_link_on_fail: bool = True,
):
    """GIF/이미지를 캐시 후 표시. 실패 시 링크 안내."""
    try:
        path = cache_remote(url, filename=filename, headers=headers, subdir=subdir)
        show_image(str(path), caption=caption)
    except Exception as e:
        if show_link_on_fail:
            st.info(f"이미지 캐시에 실패했습니다: {e}")
            st.markdown(f"[이미지 직접 열기]({url})")
        else:
            st.warning(f"이미지 표시 실패: {e}")
