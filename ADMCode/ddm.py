from __future__ import division
import numpy as np
import pandas as pd
import numba as nb
from numba.decorators import jit
from numba import float64, int64, vectorize, boolean


def gen_ddm_storage_objects(a, tr, v, z, si=.1, dt=.001, ntrials=200, deadline=1.5):

    dx = si * np.sqrt(dt)
    parameters = np.array([a, tr, v, z, si, dx, dt])

    ntime = np.int(np.floor(deadline / dt))
    rProb = np.random.random_sample((ntrials, ntime))
    traces = np.zeros_like(rProb)
    return parameters, rProb, traces



def clean_output(data, traces, deadline=1.2):
    """ create pandas dataframes from data (numpy array)
    and filter data/traces to remove failed decision trials
    ::Arguments::
    data:
    """
    # store RT/choice matrix in a pandas dataframe (DF)
    df = pd.DataFrame(data, columns=['rt', 'choice'])

    # add a column for trial number
    df.insert(0, 't', np.arange(1, 1+df.shape[0]))

    # remove trials with no boundary crossing
    df = df[(df.rt>0)&(df.rt<deadline)]

    # remove traces from failed decision trials
    traces = traces[df.index.values, :]

    return df, traces


@jit(nb.typeof((1.0, 1.0))(float64[:], float64[:], float64[:]), nopython=True)
def sim_ddm_trace(rProb, trace, parameters):

    a, tr, v, z, si, dx, dt = parameters

    vProb = .5 * (1 + (v * np.sqrt(dt))/si)
    zStart = z * a

    evidence = zStart
    trace[0] = evidence
    deadline = trace.size

    for nsteps in range(1, deadline):
        if rProb[nsteps] < vProb:
            evidence += dx
        else:
            evidence -= dx
        trace[nsteps] = evidence

        if evidence >= a:
            return tr + (nsteps * dt), 1.0
        elif evidence <= 0:
            return tr + (nsteps * dt), 0.0

    return -1.0, -1.0


@jit((float64[:], float64[:,:], float64[:,:], float64[:,:]), nopython=True)
def sim_ddm_trials(parameters, data, rProb, traces):

    ntrials = data.shape[0]
    for t in range(ntrials):
        data[t, :] = sim_ddm_trace(rProb[t], traces[t], parameters)
