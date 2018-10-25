from __future__ import division
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from numpy.random import sample as rs
from numpy import hstack as hs
from numpy import newaxis as na
from scipy.stats.distributions import norm, uniform
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
import seaborn as sns

sns.set(style='white', font_scale=1.8)
clrs = ['#3778bf', '#e74c3c', '#9b59b6', '#319455', '#feb308', '#fd7f23']

def LCA_Model(I1=10, I2=8, I0=2, k=5, B=5, si=1., Z=1, onmax=800, dt=.01, tau=.1, tmax=1.5):

    timepoints = np.arange(0, tmax, dt)
    ntime = timepoints.size

    y1 = np.zeros(ntime)
    y2 = np.zeros(ntime)
    dx=np.sqrt(si*dt/tau)

    E1=si*np.sqrt(dt/tau)*rs(ntime)
    E2=si*np.sqrt(dt/tau)*rs(ntime)

    onset=100
    for i in range(onset, ntime):
        y1[i] = y1[i-1] + (I1 + -k*y1[i-1] + -B*y2[i-1]) * dt/tau + E1[i]
        y2[i] = y2[i-1] + (I2 + -k*y2[i-1] + -B*y1[i-1]) * dt/tau + E2[i]
        y_t = np.array([y1[i], y2[i]])

        if np.any(y_t>=Z):
            rt = i; act = np.argmax(y_t)
            return y1[:i], y2[:i], rt, act
    return y1[:i], y2[:i], np.nan, np.nan


def attractor_network(I1=6, I2=3, I0=2, k=.85, B=.28, si=.3, rmax=50, b=30, g=9, Z=20, onmax=800, dt=.001, tau=.05, tmax=1.5):

    timepoints = np.arange(0, tmax, dt)
    ntime = timepoints.size

    r1 = np.zeros(ntime)
    r2 = np.zeros(ntime)
    dv = np.zeros(ntime)

    NInput = lambda x, r: rmax/(1+np.exp(-(x-b)/g))-r
    dspace = lambda r1, r2: (r1-r2)/np.sqrt(2)

    E1=si*np.sqrt(dt/tau)*rs(ntime)
    E2=si*np.sqrt(dt/tau)*rs(ntime)

    onset=100
    r1[:onset], r2[:onset] = [v[0][:onset] + I0+v[1][:onset] for v in [[r1,E1],[r2,E2]]]

    subZ=True
    for i in range(onset, ntime):
        r1[i] = r1[i-1] + dt/tau * (NInput(I1 + I0 + k*r1[i-1] + -B*r2[i-1], r1[i-1])) + E1[i]
        r2[i] = r2[i-1] + dt/tau * (NInput(I2 + I0 + k*r2[i-1] + -B*r1[i-1], r2[i-1])) + E2[i]
        dv[i] = (r1[i]-r2[i])/np.sqrt(2)
        if np.abs(dv[i])>=Z:
            rt = i+1
            return r1[:i+1], r2[:i+1], dv[:i+1], rt
    rt = i+1
    return r1[:i], r2[:i], dv[:i], rt



def simulate_attractor_competition(Imax=12, I0=0.05, k=1.15, B=.6, g=15, b=30, rmax=100, si=6.5, dt=.002, tau=.075, Z=100, ntrials=250):

    sns.set(style='white', font_scale=1.8)
    f, ax = plt.subplots(1, figsize=(8,7))
    cmap = mpl.colors.ListedColormap(sns.blend_palette([clrs[1], clrs[0]], n_colors=ntrials))
    Iscale = np.hstack(np.tile(np.linspace(.5*Imax, Imax, ntrials/2)[::-1], 2))
    Ivector=np.linspace(-1,1,len(Iscale))
    norm = mpl.colors.Normalize(
        vmin=np.min(Ivector),
        vmax=np.max(Ivector))
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    for i, I_t in enumerate(Iscale):
        if i < (ntrials/2.):
            I1 = Imax; I2 = I_t
        else:
            I1=I_t; I2 = Imax
        r1, r2, dv, rt = attractor_network(I1=I1, I2=I2, I0=I0, k=k, B=B, g=g, b=b, rmax=rmax, si=si, dt=dt, tau=tau, Z=Z)
        ax.plot(r1, r2, color=sm.to_rgba(Ivector[i]), alpha=.5)

    c_ax = plt.colorbar(sm, ax=plt.gca())
    c_ax.set_ticks([-1, 1])
    c_ax.set_ticklabels(['$I_1<<I_2$', '$I_1>>I_2$'])
    ax.plot([0,rmax], [0,rmax], color='k', alpha=.5, linestyle='-', lw=3.5)
    _=plt.setp(ax, ylim=[0,rmax], xlim=[0,rmax], xticks=[0,rmax], xticklabels=[0,rmax],
               yticks=[0,rmax],yticklabels=[0,rmax], ylabel='$r_1$ (Hz)', xlabel='$r_2$ (Hz)')


def simulate_attractor_behavior(I1=12, I2=9, I0=0.05, k=1.15, B=1., g=12, b=35, rmax=100, si=5., dt=.001, tau=.075, Z=30, ntrials=250):

    behavior = np.zeros((ntrials, 3))
    for t in range(ntrials):
        r1, r2, dv, rt = attractor_network(I1=I1, I2=I2, I0=I0, k=k, B=B, g=g, b=b, rmax=rmax,  si=si, dt=dt, tau=tau, Z=Z)

        choice=0
        acc=0
        if dv[-1]>=Z:
            choice=1
            acc=0
            if I1>I2: acc=1
        elif dv[-1]<=-Z:
            choice=2
            if I2>I1: acc=1
        elif I2==I1:
            acc=.5


        behavior[t, :] = choice, acc, rt

    return pd.DataFrame(behavior, columns=['choice', 'accuracy', 'rt'], index=np.arange(ntrials))


def SAT_experiment(dfa, dfb):

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    dfa['cond']='Control'
    dfb['cond'] = 'Test'
    dfx = pd.concat([dfa,dfb])
    dfacc = dfx[dfx.accuracy==1]

    accY=dfx.groupby('cond').mean()['accuracy'].values
    ax1.scatter([0], accY[0], s=205, color='k', alpha=1.)
    ax1.scatter([1], accY[1], s=205, color=clrs[0], alpha=1.)
    ax1.plot([0,1], accY, color='k', alpha=.3, linewidth=3.5)

    sns.kdeplot(dfacc[dfacc.cond=='Control'].rt.values, ax=ax2, shade=True, color='k', alpha=.15, lw=0)
    sns.kdeplot(dfacc[dfacc.cond=='Test'].rt.values, ax=ax2, shade=True, color=clrs[0], linewidth=0)

    rtmu = dfacc.groupby('cond').mean()['rt'].values
    xmax = ax2.get_ylim()[-1]
    ax2.vlines(rtmu[0], 0, xmax, color='k', linestyles='--', linewidth=2, label='Control')
    ax2.vlines(rtmu[1], 0, xmax, color=clrs[0], linewidth=2, label='Test')
    ax2.set_yticklabels([])
    ax1.set_ylim(0,1)
    ax1.set_xlim(-.5,1.5)
    ax1.set_xticks([0,1])
    ax1.set_xticklabels(['Control','Test'])
    ax1.set_ylabel('% Correct')
    ax1.set_xlabel('Condition')
    ax2.set_xlabel('RT (ms)')
    ax2.legend()
    sns.despine()

def noisy_attractor_endpoints(I=12, I0=0.05, k=1.15, B=1.15, g=25, b=50, rmax=100, si=6.5, dt=.002, tau=.05, Z=100, ntrials=250):

    f, axes = plt.subplots(1, 4, figsize=(12,3.5))

    for i in range(4):
        attractor_endpoints(I=I[i], I0=I0, k=k, B=B, g=g, b=b, rmax=rmax, si=si, dt=dt, tau=tau, Z=Z, ntrials=ntrials, ax=axes[i])
        if i>0:
            axes[i].set_yticklabels([])
            axes[i].set_ylabel('')
        else:
            axes[i].set_yticks([0, 120])
            axes[i].set_yticklabels([0, 120])
            axes[i].set_ylim(0,120)
    plt.tight_layout()


def attractor_endpoints(I=12, I0=0.05, k=1.15, B=1.15, g=25, b=50, rmax=100, si=6.5, dt=.002, tau=.05, Z=100, ntrials=250, ax=None):

    sns.set(style='white', font_scale=1.8)
    if ax is None:
        f, ax = plt.subplots(1, figsize=(4,4))

    r1d1,r2d1,r1d2,r2d2 = [],[],[],[]
    for i in range(ntrials):
        r1, r2, dv, rt = attractor_network(I1=I, I2=I, I0=I0, k=k, B=B, g=g, b=b, rmax=rmax, si=si, dt=dt, tau=tau, Z=Z)
        if r1[-1]>r2[-1]:
            r1d1.append(r1[-1])
            r2d1.append(r2[-1])
        if r2[-1]>r1[-1]:
            r1d2.append(r1[-1])
            r2d2.append(r2[-1])

    ax.scatter(r2d1, r1d1, s=30, color=clrs[0], marker='o', alpha=.1)
    ax.scatter(r2d2, r1d2, s=30, color=clrs[1], marker='o', alpha=.1)

    #xymax = np.max(np.hstack([r1d1, r2d1]))
    #xymax = np.max(np.hstack([r1d2, r2d2]))
    xymax=120
    rmax=int(xymax)
    ax.plot([0,xymax], [0,xymax], color='k', alpha=.5, linestyle='-', lw=3.5)
    _ = plt.setp(ax, ylim=[0,xymax], xlim=[0,xymax], xticks=[0,xymax], xticklabels=[0,rmax], yticks=[0,xymax],yticklabels=[0,rmax], ylabel='$r_1$ (Hz)', xlabel='$r_2$ (Hz)')


def plot_sigmoid_response(b=50, g=20, rmax=1):

    x = np.linspace(0,100,100)
    y = rmax/(1+np.exp(-(x-b)/g))

    plt.vlines(b, 0, y[b], color='r', label='b')
    plt.hlines(.5, 0, b, color='k', linestyles='--')
    plt.fill_between(x[:b+1], 0, y[:b+1], alpha=.05, color='k')
    plt.text(b+2, .045, 'b', color='r')

    # plot g slope
    w_lo = int(x[b])
    w_hi = int(x[b+10])
    plt.plot([w_lo, w_hi], [y[w_lo]+.03, y[w_hi]+.03], color='b')
    plt.text(b, y[b+5]+.04, 'g', color='b')

    # plot f-i curve
    plt.plot(x, y, color='k')

    ax = plt.gca()
    ax.set_xlabel('Input Current')
    ax.set_ylabel('Neural Response')
    ax.set_xticks([0,100])
    ax.set_xlim(0,100)
    ax.set_xticklabels([0,100])
    ax.set_ylim(0, rmax*1.05)
    sns.despine()


def plot_decision_dynamics(r1, r2, dv, Z=20, axes=None, alpha=.7, label=None, xlim=None):

    if axes is None:
        f, axes = plt.subplots(2, 1, figsize=(6,9))
    ax2, ax1 = axes
    rt=len(dv)-1

    l1, l2, l3 = [None]*3

    if label:
        l1, l2, l3 = '$y_1$', '$y_2$', '$\Delta y$'
        ylabel = 'Activation'
    ax1.plot(r1, color=clrs[0], label=l1, linewidth=2.5, alpha=alpha)
    ax1.plot(r2, color=clrs[1],  label=l2, linewidth=2.5, alpha=alpha)
    ax1.vlines(rt, ymin=r2[rt], ymax=r1[rt], color=clrs[0], linestyles='--', alpha=alpha)
    ax2.plot(dv, color=clrs[2], label=l3, linewidth=2.5, alpha=alpha)

    if xlim is None:
        xlim = ax1.get_xlim()

    xmin, xmax = xlim
    for ax in [ax1,ax2]:
        ax.set_xlim(xmin, xmax)
        ax.legend(loc=2)
    ax2.set_yticklabels([])
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel(ylabel)
    ax2.set_ylabel('Decision State')
    ax2.set_ylim(-Z, Z)
    ax2.hlines(0, xmin=0, xmax=xmax, color='k', linestyles='--', alpha=.5)
    ax2.hlines(Z-.25, 0, xmax, color=clrs[2], alpha=1., linestyles='-', lw=4)
    ax2.hlines(-Z+.25, 0, xmax, color=clrs[2], alpha=1., linestyles='-', lw=4)
    ax2.set_xticklabels([])
    sns.despine(ax=ax1)

    sns.despine(ax=ax2, right=True, top=True, bottom=True)

def plot_rt_distributions(ax1, ax2, rts, xlim=None, alpha=.8):

    divider = make_axes_locatable(ax2)
    axx = divider.append_axes("top", size=1.6, pad=0, sharex=ax2)
    for rt in rts:
        sns.kdeplot(rt, ax=axx, shade=True, color=clrs[2], alpha=alpha)
        alpha=alpha-.5

    for spine in ['top', 'left', 'bottom', 'right']:
        axx.spines[spine].set_visible(False)

    axx.set_xticklabels([])
    axx.set_yticklabels([])
    ax2.set_yticklabels([])
    xmin, xmax = ax1.get_xlim()
    ax1.set_xlim(0, xmax)
    Z = ax2.get_ylim()[-1]

    ax2.hlines(0, xmin=0, xmax=xmax, color='k', linestyles='--', alpha=.5)
    ax2.hlines(Z-.25, 0, xmax, color=clrs[2], alpha=1., linestyles='-', lw=4)
    ax2.hlines(-Z+.25, 0, xmax, color=clrs[2], alpha=1., linestyles='-', lw=4)
