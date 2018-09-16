import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.signal import detrend


def sdt_mle(h, m, cr, fa):

    """Calculates maximum-likelihood estimates of sensitivity and bias.

    Args:
        h: Hits
        m: Misses
        cr: Correct Rejections.
        fa: False Alarms

    Returns:
        d (d-prime)
        c (criterion)

    """

    H, M, CR, FA, = h, m, cr, fa

    n0, n1 = float(FA + CR), float(H + M)
    if H == 0:  H += 0.5
    if H == n1: H -= 0.5
    if FA == 0: FA += 0.5
    if FA == n0: FA -= 0.5

    pH = H / float(n1)
    pFA = FA / float(n0)
    d = norm.ppf(pH) - norm.ppf(pFA)
    c = -0.5 * (norm.ppf(pH) + norm.ppf(pFA))

    return d, c
