# utils/latex.py
from __future__ import annotations
import numpy as np

__all__ = [
    "bmatrix", "colvec", "vector",
    "axb", "system", "lincomb"
]

# 숫자 포맷(유효숫자 g, -0 방지)
def _fmt(x, digits=3):
    try:
        xv = float(x)
    except Exception:
        return str(x)
    s = f"{xv:.{digits}g}"
    return "0" if s == "-0" else s

def bmatrix(M, digits=3) -> str:
    rows = [" & ".join(_fmt(x, digits) for x in row) for row in M]
    return r"\begin{bmatrix}" + r"\\ ".join(rows) + r"\end{bmatrix}"

def colvec(v, digits=3) -> str:
    return r"\begin{bmatrix}" + r"\\ ".join(_fmt(x, digits) for x in v) + r"\end{bmatrix}"

def vector(name: str, v, digits=3) -> str:
    return rf"{name}={colvec(v, digits)}"

def axb(A, b, with_labels=True, det=None, residual=None, digits=3) -> str:
    Atex = bmatrix(A, digits)
    xtex = r"\begin{bmatrix}x_1\\x_2\\x_3\end{bmatrix}"
    btex = colvec(b, digits)
    if with_labels:
        expr = rf"\underbrace{{{Atex}}}_A\;\underbrace{{{xtex}}}_x\;=\;\underbrace{{{btex}}}_b"
    else:
        expr = rf"{Atex}\,{xtex}\;=\;{btex}"
    if det is not None:
        expr += rf"\qquad \det(A)={_fmt(det, digits)}"
    if residual is not None:
        expr += rf"\quad \|Ax-b\|_2={_fmt(residual, digits)}"
    return expr

def system(A, b, digits=3) -> str:
    (a11,a12,a13),(a21,a22,a23),(a31,a32,a33) = A
    return rf"""
\begin{{cases}}
{_fmt(a11,digits)}x_1 + {_fmt(a12,digits)}x_2 + {_fmt(a13,digits)}x_3 = {_fmt(b[0],digits)}\\
{_fmt(a21,digits)}x_1 + {_fmt(a22,digits)}x_2 + {_fmt(a23,digits)}x_3 = {_fmt(b[1],digits)}\\
{_fmt(a31,digits)}x_1 + {_fmt(a32,digits)}x_2 + {_fmt(a33,digits)}x_3 = {_fmt(b[2],digits)}
\end{{cases}}
"""

def lincomb(name: str, coeffs, basis=("v_1","v_2","v_3"), digits=3) -> str:
    terms = [rf"{_fmt(c, digits)}\,{b}" for c, b in zip(coeffs, basis)]
    return rf"{name}=" + " + ".join(terms)
