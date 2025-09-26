# utils/frac.py
def _gcd(a:int, b:int)->int:
    a, b = abs(a), abs(b)
    while b: a, b = b, a % b
    return max(a, 1)

def _lcm(a:int, b:int)->int:
    return abs(a*b) // _gcd(a,b)

def simplify(n:int, d:int):
    if d == 0: return n, d
    g = _gcd(n, d)
    n //= g; d //= g
    if d < 0: n, d = -n, -d
    return n, d

def add_fractions(n1,d1,n2,d2):
    L = _lcm(int(d1), int(d2))
    n = int(n1)*(L//int(d1)) + int(n2)*(L//int(d2))
    return simplify(n, L)

def to_mixed(n:int, d:int):
    if d == 0: return None
    q, r = divmod(abs(n), d)
    sgn = -1 if n<0 else 1
    return sgn*q, r, d
