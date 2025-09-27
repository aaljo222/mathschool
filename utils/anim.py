# utils/anim.py
import time

def step_loop(n: int, fps: int = 20, key: str = "loop"):
    """프레임 0..n-1을 생성하는 간단한 루프 제너레이터"""
    delay = 1.0 / max(1, fps)
    for i in range(n):
        time.sleep(delay)
        yield i

# (하위호환) 일부 탭이 아직 playbar를 import 할 수도 있으니 안전하게 더미 제공
def playbar(*args, **kwargs):
    return False  # 항상 정지 상태로 반환
