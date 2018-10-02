#!usr/bin/env python
from __future__ import division
import pandas as pd
import numpy as np
from future.utils import listvalues


def analyze_choice_behavior(df, nblocks=10):
    subject_choices = lambda idf: idf.choice.value_counts()/idf.shape[0]
    bdf = blockify_trials(df, groups=['agent'], nblocks=nblocks)
    xdf = bdf.groupby('agent').apply(subject_choices).reset_index()
    muChoice = xdf.sort_values(['agent', 'level_1']).groupby('level_1').mean().choice.values
    errChoice = xdf.sort_values(['agent', 'level_1']).groupby('level_1').sem().choice.values * 1.96
    return muChoice, errChoice

def blockify_trials(data, nblocks=5, conds=None, groups=['agent']):

    datadf = data.copy()
    if conds is not None:
        if type(conds) is str:
            conds = [conds]
        groups = groups + conds

    idxdflist = []
    for dfinfo, idxdf in datadf.groupby(groups):
        ixblocks = np.array_split(idxdf.trial.values, nblocks)
        blocks = np.hstack([[i+1]*arr.size for i, arr in enumerate(ixblocks)])
        idxdf = idxdf.copy()
        colname = 'block'
        idxdf[colname] = blocks
        idxdflist.append(idxdf)

    return pd.concat(idxdflist)
