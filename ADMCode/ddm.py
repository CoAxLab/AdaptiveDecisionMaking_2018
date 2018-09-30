from __future__ import division
import numpy as np
import pandas as pd
import numba as nb
from numba.decorators import jit
from numpy.random import random_sample
from numba import float64, int64, vectorize, boolean


def sim_ddm_trials(parameters, ntrials=500, deadline=1.5):
    """ main user interface function for simulating multiple trials
    with the DDM (wrapper for numba compiled _sim_ddm_trials_ func)

    :: Arguments ::
        parameters: 1d array (Nparams) of DDM parameters
            a: boundary height
            tr: non-decision time
            v: drift-rate
            z: starting point (frac of a; 0.0 < z < 1.0)
            si: diffusion constant (sigma param in DDM equation from lecture)
            dx: step-size of evidence
            dt: time step
        ntrials : (int) number of trials to simulate
        deadline (max time for accumualtion in seconds)

    :: Returns ::
        data (pd.DataFrame): pandas DF with rt and choice data
        traces (np.ndarray): 2d array (Ntrials x Ntime) of evidence traces
    """

    # generate storage objects for data/traces
    data, rProb, traces = gen_ddm_storage_objects(parameters, ntrials, deadline)

    # simulate ntrials w/ DDM and fill data & traces array
    _sim_ddm_trials_(parameters, data, rProb, traces)

    # filter data/traces and return as pd.DataFrame
    df, traces = clean_output(data, traces, deadline=deadline)
    return df, traces



@jit(nb.typeof((1.0, 1.0))(float64[:], float64[:], float64[:]), nopython=True)
def sim_ddm(parameters, rProb, trace):
    """ single trial simulation of the DDM (discrete time / random walk)

    ::Arguments::
        parameters: 1d array (Nparams) of DDM parameters
        rProb: 1d array (Ntimesteps) of random floats between 0 and 1
        trace: 1d array (Ntimesteps) for storing the evidence trace

    ::Returns::
        RT (float): the time that evidence crossed one of the boundaries
        choice: 1 if evidence terminated at upper bound, 0 if lower bound
    """

    # extract parameters
    a, tr, v, z, si, dx, dt = parameters

    # convert drift-rate into a probability,
    # & scale by sigma (si) and timestep (dt)
    # if v > 0, then 0.5 < vProb < 1.0
    # if v < 0, then 0.0 < vProb < 0.5
    vProb = .5 * (1 + (v * np.sqrt(dt))/si)

    # define starting point with respect to boundary height
    zStart = z * a

    #initialize evidence variable at zStart
    evidence = zStart
    trace[0] = evidence

    # define deadline (max time allowed for accumulation)
    deadline = trace.size

    for nsteps in range(1, deadline):
        # sample a random probability (r) and compare w/ vProb
        if rProb[nsteps] < vProb:
            # if r < vProb, step evidence up (towards a)
            evidence += dx
        else:
            # if r > vProb, step evidence down (towards 0)
            evidence -= dx
        # store new value of evidence at current timestep
        trace[nsteps] = evidence

        # check if new value of evidence crossed bound
        if evidence >= a:
            # calculate RT (in milliseconds)
            rt = tr + (nsteps * dt)
            # set choice to 1.0 (upper bound)
            choice = 1.0

            # terminate simulation, return rt & choice
            return rt, choice

        elif evidence <= 0:
            # calculate RT (in milliseconds)
            rt = tr + (nsteps * dt)
            # set choice to 0.0 (lower bound)
            choice = 0.0

            # terminate simulation, return rt & choice
            return rt, choice

    # return -1.0 for rt and choice so we can filter out
    # trials where evidence never crossed 0 or a
    return -1.0, -1.0



@jit((float64[:], float64[:,:], float64[:,:], float64[:,:]), nopython=True)
def _sim_ddm_trials_(parameters, data, rProb, traces):
    """ called by sim_ddm_trials() func to speed up trial iteraion
    """
    ntrials = data.shape[0]
    for t in range(ntrials):
        data[t, :] = sim_ddm(parameters, rProb[t], traces[t])


def gen_ddm_storage_objects(parameters, ntrials=200, deadline=1.5):
    """ create pandas dataframes from data (numpy array)
    and filter data/traces to remove failed decision trials
    ::Arguments::
        parameters (array): 1d array (Nparams) of DDM parameters
        ntrials : (int) number of trials to simulate
        deadline (float): (max time for accumualtion in seconds)

    ::Returns::
        data (ndarray): ndarray with rt and choice data
        rProb (ndarray): 2d array (Ntrials x Ntimesteps) w. random floats (0-1)
        traces (ndarray): 2d array (Ntrials x Ntime) of evidence traces
    """
    dt = parameters[-1]
    ntime = np.int(np.floor(deadline / dt))

    # empty matrix Ntrials x 2 (cols for RT & Choice)
    data = np.zeros((ntrials, 2))
    # 1d array (Ntimesteps) of random floats between 0 and 1
    rProb = random_sample((ntrials, ntime))
    # 1d array (Ntimesteps) for storing evidence traces
    traces = np.zeros_like(rProb)
    return data, rProb, traces


def clean_output(data, traces, deadline=1.2, stimulus=None):
    """ create pandas dataframes from data (numpy array)
    and filter data/traces to remove failed decision trials
    ::Arguments::
        data (ndarray): ndarray with rt and choice data
        traces (ndarray): 2d array (Ntrials x Ntime) of evidence traces
    ::Returns::
        data (pd.DataFrame): pandas DF with rt and choice data
        traces (ndarray): 2d array (Ntrials x Ntime) filtered traces
    """
    # store RT/choice matrix in a pandas dataframe (DF)
    df = pd.DataFrame(data, columns=['rt', 'choice'])

    # add a column for trial number
    df.insert(0, 'trial', np.arange(1, 1+df.shape[0]))

    # remove trials with no boundary crossing
    df = df[(df.rt>0)&(df.rt<deadline)]

    # remove traces from failed decision trials
    traces = traces[df.index.values, :]

    df = df.reset_index(drop=True)

    return df, traces



def ddm_sim_yesno(param_list, ntrials, deadline):

    psignal, pnoise = param_list
    sDF, sTraces = sim_ddm_trials(psignal, ntrials, deadline)
    nDF, nTraces = sim_ddm_trials(pnoise, ntrials, deadline)

    sDF['stim'] = 'signal'
    nDF['stim'] = 'noise'

    sdt_vals = dict(zip(['H', 'M', 'CR', 'FA'], np.zeros(4).astype(int)))

    dflist=[]
    for dfi in [sDF, nDF]:
        sdtDF = pd.DataFrame(sdt_vals, index=dfi.index)
        dfi = pd.concat([dfi, sdtDF], axis=1)

        if dfi.loc[0, 'stim']=='signal':
            dfi.loc[dfi.choice==1, 'H'] = 1
            dfi.loc[dfi.choice==0, 'M'] = 1
        else:
            dfi.loc[dfi.choice==0, 'CR'] = 1
            dfi.loc[dfi.choice==1, 'FA'] = 1

        dflist.append(dfi)

    df = pd.concat(dflist).reset_index(drop=True)

    return df, [sTraces, nTraces]
