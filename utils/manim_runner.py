from pathlib import Path
from manim import config, Scene
from typing import Type

def render_manim(scene_cls: Type[Scene], cache_key: str,
                 w=640, h=360, fps=20, media_dir="manim_cache") -> Path:
    out_root = Path(media_dir) / cache_key
    out_root.mkdir(parents=True, exist_ok=True)

    # 경량 설정
    config.pixel_width = w
    config.pixel_height = h
    config.frame_rate = fps
    config.quality = "low_quality"     # 품질 낮춤
    config.media_dir = str(out_root)
    config.write_to_movie = True
    config.save_last_frame = False
    config.disable_caching = True
    config.tex_template = None         # LaTeX 회피

    scene_cls().render()
    mp4s = sorted(out_root.rglob("*.mp4"), key=lambda p: p.stat().st_mtime)
    if not mp4s:
        raise RuntimeError("No video rendered")
    return mp4s[-1]
