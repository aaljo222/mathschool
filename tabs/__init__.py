# tabs/__init__.py
from . import tab_conic, tab_trig, tab_calc_def, tab_linreg, tab_taylor, tab_fft, tab_euler
try:
    from . import tab_manim  # noqa: F401
except Exception:
    pass
__all__ = [
    "tab_conic",
    "tab_trig",
    "tab_calc_def",
    "tab_linreg",
    "tab_taylor",
    "tab_fft",
    "tab_euler",
]
