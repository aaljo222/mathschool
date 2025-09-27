# utils/anim.py

import time

def step_loop(n: int, fps: int = 20, key: str = "loop"):
    delay = 1.0 / max(1, fps)
    for i in range(n):
        time.sleep(delay)
        yield i

# ✅ 호환용: 예전 코드가 기대하는 next_frame_index 이름을 제공
def next_frame_index(total_frames: int, fps: int = 20, key: str = "loop"):
    """Yield frame indices 0..total_frames-1 at the given fps (compat shim)."""
    yield from step_loop(total_frames, fps=fps, key=key)

# (옵션) 하위호환 더미
def playbar(*args, **kwargs):
    return False
