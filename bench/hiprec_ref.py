"""40-digit Decimal references for lgamma and digamma (x > 0), independent of scipy and libm.

Shift x up to y >= 30 with the recurrences, then use the Stirling / asymptotic series with
exact Bernoulli numbers through B_40. Truncation error at y >= 30 is below 1e-40.
"""
from decimal import Decimal, getcontext
from fractions import Fraction

getcontext().prec = 45


def _bernoulli(nmax):
    # Akiyama-Tanigawa; returns B_0..B_nmax as Fractions (B_1 = +1/2 convention unused here)
    B = []
    a = [Fraction(0)] * (nmax + 1)
    for m in range(nmax + 1):
        a[m] = Fraction(1, m + 1)
        for j in range(m, 0, -1):
            a[j - 1] = j * (a[j - 1] - a[j])
        B.append(a[0])
    return B


_B = _bernoulli(40)
_B2K = [(k, Decimal(_B[2 * k].numerator) / Decimal(_B[2 * k].denominator)) for k in range(1, 21)]
_HALF_LN_2PI = (Decimal(2) * Decimal("3.14159265358979323846264338327950288419716939937510")).ln() / 2
SHIFT = 30


def lgamma_dec(x: float) -> float:
    y = Decimal(x)
    acc = Decimal(0)
    while y < SHIFT:          # lgamma(y) = lgamma(y + 1) - ln(y)
        acc += y.ln()
        y += 1
    s = (y - Decimal("0.5")) * y.ln() - y + _HALF_LN_2PI
    for k, b in _B2K:
        s += b / (2 * k * (2 * k - 1) * y ** (2 * k - 1))
    return float(s - acc)


def psi_dec(x: float) -> float:
    y = Decimal(x)
    acc = Decimal(0)
    while y < SHIFT:          # psi(y) = psi(y + 1) - 1/y
        acc += 1 / y
        y += 1
    s = y.ln() - 1 / (2 * y)
    for k, b in _B2K:
        s -= b / (2 * k * y ** (2 * k))
    return float(s - acc)


_PI = Decimal("3.14159265358979323846264338327950288419716939937510")


def _sincos_pi(x: Decimal):
    """sin(pi x), cos(pi x) by reducing x mod 2 exactly, then Taylor series."""
    r = x % 2
    if r > 1:
        r -= 2
    t = _PI * r
    s, c, term_s, term_c = Decimal(0), Decimal(0), t, Decimal(1)
    for n in range(60):
        s += term_s
        c += term_c
        term_s *= -t * t / ((2 * n + 2) * (2 * n + 3))
        term_c *= -t * t / ((2 * n + 1) * (2 * n + 2))
    return s, c


def _psi_dec_pos(y):
    acc = Decimal(0)
    while y < SHIFT:
        acc += 1 / y
        y += 1
    s = y.ln() - 1 / (2 * y)
    for k, b in _B2K:
        s -= b / (2 * k * y ** (2 * k))
    return s - acc


def _lgamma_dec_pos(y):
    acc = Decimal(0)
    while y < SHIFT:
        acc += y.ln()
        y += 1
    s = (y - Decimal("0.5")) * y.ln() - y + _HALF_LN_2PI
    for k, b in _B2K:
        s += b / (2 * k * (2 * k - 1) * y ** (2 * k - 1))
    return s - acc


def psi_dec_any(x: float) -> float:
    """Digamma for any non-pole x: reflection psi(x) = psi(1 - x) - pi cos(pi x)/sin(pi x)."""
    d = Decimal(x)
    if d > 0:
        return float(_psi_dec_pos(d))
    s, c = _sincos_pi(d)
    return float(_psi_dec_pos(1 - d) - _PI * c / s)


def lgamma_dec_any(x: float) -> float:
    """log|Gamma(x)| for any non-pole x: lgamma(x) = ln(pi / |sin(pi x)|) - lgamma(1 - x)."""
    d = Decimal(x)
    if d > 0:
        return float(_lgamma_dec_pos(d))
    s, _ = _sincos_pi(d)
    return float((_PI / abs(s)).ln() - _lgamma_dec_pos(1 - d))


def erfc_dec(x: float) -> float:
    """erfc for x >= 1 via the Laplace continued fraction, evaluated backward in Decimal.

    erfc(x) = exp(-x^2)/sqrt(pi) * 1/(x + (1/2)/(x + 1/(x + (3/2)/(x + 2/(x + ...)))))
    """
    d = Decimal(x)
    assert d >= 1
    f = d
    for n in range(400, 0, -1):
        f = d + Decimal(n) / 2 / f
    return float((-(d * d)).exp() / _PI.sqrt() / f)


def erfc_dec_any(x: float) -> float:
    """erfc for any x: continued fraction for |x| >= 1, Taylor series of erf for |x| < 1,
    and erfc(-x) = 2 - erfc(x)."""
    d = Decimal(x)
    if d >= 1:
        return erfc_dec(x)
    if d <= -1:
        return float(2 - Decimal(erfc_dec(-x)))  # erfc(-x) in (1, 2): absolute error is tiny
    s, term, n = Decimal(0), d, 0
    while abs(term) > Decimal(10) ** -44:
        s += term / (2 * n + 1)
        n += 1
        term *= -d * d / n
    return float(1 - 2 / _PI.sqrt() * s)
