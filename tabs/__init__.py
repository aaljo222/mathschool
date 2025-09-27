# tabs/__init__.py
from . import (
    tab_conic, tab_trig, tab_calc_def, tab_linreg, tab_taylor, tab_fft, tab_euler,
    tab_exp_log, tab_log_add,          # <-- 로그-덧셈 탭 포함
    tab_basics, tab_lincomb3, tab_circuits,
    tab_newton, tab_grad_descent, tab_fourier_series, tab_convolution,
    tab_svd, tab_eigen2d, tab_bezier, tab_lissajous, tab_wave_interference, tab_markov,
)

__all__ = [
    "tab_conic", "tab_trig", "tab_calc_def", "tab_linreg", "tab_taylor", "tab_fft", "tab_euler",
    "tab_exp_log", "tab_log_add",
    "tab_basics", "tab_lincomb3", "tab_circuits",
    "tab_newton", "tab_grad_descent", "tab_fourier_series", "tab_convolution",
    "tab_svd", "tab_eigen2d", "tab_bezier", "tab_lissajous", "tab_wave_interference", "tab_markov",
]
