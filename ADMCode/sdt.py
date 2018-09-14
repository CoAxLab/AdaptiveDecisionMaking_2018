import numpy as np
import pandas as pd
import patsy
import pymc3 as pm
import theano
import theano.tensor as T
from scipy.stats import norm
from scipy.signal import detrend


def sdt_mle(f, h, m, r):
    """Calculates maximum-likelihood estimates of sensitivity and bias.

    Args:
        f: False alarms.
        h: Hits.
        m: Misses.
        r: Correct rejections.

    Returns:
        [(d1, c1) ...]

    """
    out = []
    for _f, _h, _m, _r in zip(f, h, m, r):

        n0, n1 = float(_f + _r), float(_h + _m)
        if _f == 0:
            _f += 0.5
        if _f == n0:
            _f -= 0.5
        if _h == 0:
            _h += 0.5
        if _h == n1:
            _h -= 0.5

        fhat = _f / float(n0)
        hhat = _h / float(n1)
        d = norm.ppf(hhat) - norm.ppf(fhat)
        c = -0.5 * (norm.ppf(hhat) + norm.ppf(fhat))
        out.append((d, c))
    return out
