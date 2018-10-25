from __future__ import division
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from ADMCode import sdt, utils
import matplotlib.pyplot as plt
from scipy.stats.stats import sem

def convert_params(parameters, maxtime=1.5):
    a, tr, v, z, si, dx, dt = parameters
    zStart = z * a
    trSteps = int(tr/dt)
    deadline = (maxtime / dt) * 1.1
    return a, trSteps, v, zStart, si, dx, dt, deadline


def build_ddm_axis(parameters, maxtime=1.5):

    sns.set(style='white')
    f, ax = plt.subplots(1, figsize=(8.5, 7), sharex=True)

    a, tr, v, z, si, dx, dt, deadline = convert_params(parameters, maxtime)
    w = deadline
    xmin=tr - 100

    plt.setp(ax, xlim=(xmin - 51, w + 1), ylim=(0 - (.01 * a), a + (.01 * a)))
    ax.hlines(y=a, xmin=xmin, xmax=w, color='#3572C6', linewidth=4)
    ax.hlines(y=0, xmin=xmin, xmax=w, color='#e5344a', linewidth=4)
    ax.hlines(y=z, xmin=xmin, xmax=w, color='k', alpha=.4, linestyles='--', linewidth=3)
    ax.vlines(x=xmin-50, ymin=-.1, ymax=a+.1, color='k', alpha=.15, linewidth=5)
    ax.hlines(y=z, xmin=xmin, xmax=tr, color='k', linewidth=4)

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    sns.despine(top=True, right=True, bottom=True, left=True, ax=ax)

    divider = make_axes_locatable(ax)
    axx1 = divider.append_axes("top", size=1.2, pad=0.0, sharex=ax)
    axx2 = divider.append_axes("bottom", size=1.2, pad=0.0, sharex=ax)
    plt.setp(axx1, xlim=(xmin - 51, w + 1))#ylim=(0 - (.01 * a), a + (.01 * a)))
    plt.setp(axx2, xlim=(xmin - 51, w + 1))#ylim=(0 - (.01 * a), a + (.01 * a)))
    axx2.invert_yaxis()
    axx1.hist([0], normed=False, bins=np.linspace(200, w, num=9), alpha=1., color='White')
    axx2.hist([0], normed=False, bins=np.linspace(200, w, num=9), alpha=1., color='White')

    for axx in [axx1, axx2]:
        for spine in ['top', 'left', 'bottom', 'right']:
            axx.spines[spine].set_visible(False)
        axx.set_xticklabels([])
        axx.set_yticklabels([])
    return f, [ax, axx1, axx2]



def plot_ddm_sims(df, parameters, traces=None, plot_v=False, fig=None, colors=None, vcolor='k', kdeplot=True):

    maxtime = df.rt.max()
    a, trSteps, v, zStart, si, dx, dt, deadline = convert_params(parameters, maxtime)

    if colors is None:
        colors = ['#3572C6', '#e5344a']
    if fig is None:
        f, axes = build_ddm_axis(parameters, maxtime)
    else:
        f = fig; axes = fig.axes

    plot_bound_rts(df, parameters, f=f, colors=colors, kdeplot=kdeplot)

    if traces is not None:
        plot_traces(df, parameters, traces, f=f, colors=colors)

    if plot_v:
        plot_drift_line(df, parameters, color=vcolor, ax=f.axes[0])

    return f


def compare_drift_effects(df, param_list):

    sDF = df[df.stim=='signal']
    nDF = df[df.stim=='noise']
    colors = [['#009e07','#009e07'], ["#e5344a", "#e5344a"]]

    maxtime = df.rt.max()
    a, trSteps, v, zStart, si, dx, dt, deadline = convert_params(param_list[0], maxtime)
    f=None
    for i, dfi in enumerate([sDF, nDF]):
        clrs = colors[i]
        f = plot_ddm_sims(dfi, param_list[i], colors=clrs, plot_v=True, fig=f, vcolor=clrs[0], kdeplot=False)

    ax, axx1, axx2 = f.axes
    xmin = trSteps-100
    ax.hlines(y=a, xmin=xmin, xmax=deadline, color='k', linewidth=4)
    ax.hlines(y=0, xmin=xmin, xmax=deadline, color='k', linewidth=4)

    if sDF.shape[0] > nDF.shape[0]:
        ymax, ymin = axx1.get_ylim()[::-1]
        axx2.set_ylim(ymax, ymin)
    else:
        ymax, ymin = axx2.get_ylim()[::-1]
        axx1.set_ylim(ymax, ymin)
    return ax


def plot_bound_rts(df, parameters, f, colors=None, kdeplot=True):

    a, trSteps, v, zStart, si, dx, dt, deadline = convert_params(parameters)
    rt1 = df[df.choice==1].rt.values / dt
    rt0 = df[df.choice==0].rt.values / dt

    if colors is None:
        colors = ['#3572C6', '#e5344a']
    ax, axx1, axx2 = f.axes
    clip = (df.rt.min()/dt, deadline)

    if kdeplot:
        sns.kdeplot(rt1, alpha=.5, linewidth=0, color=colors[0], ax=axx1, shade=True, clip=clip, bw=15)
        sns.kdeplot(rt0, alpha=.5, linewidth=0, color=colors[1], ax=axx2, shade=True, clip=clip, bw=15)

        ymax = (.005, .01)
        if rt1.size < rt0.size:
            ymax = (.01, .005)
        axx1.set_ylim(0, ymax[0])
        axx2.set_ylim(ymax[1], 0.0)
        # axx2.invert_yaxis()

    else:
        sns.distplot(rt1, color=colors[0], ax=axx1, kde=False, norm_hist=False)
        sns.distplot(rt0, color=colors[1], ax=axx2, kde=False, norm_hist=False)


def plot_traces(df, parameters, traces, f, colors):

    a, trSteps, v, zStart, si, dx, dt, deadline = convert_params(parameters)
    ax = f.axes[0]
    ntrials = int(traces.shape[0])
    for i in range(ntrials):
        trace = traces[i]
        c = colors[0]
        nsteps = np.argmax(trace[trace<=a]) + 2
        if df.iloc[i]['choice']==0:
        # if trace[nsteps]<zStart:
            c = colors[1]
            nsteps = np.argmin(trace[trace>=0]) + 2
        ax.plot(np.arange(trSteps, trSteps + nsteps), traces[i, :nsteps], color=c, alpha=.1)


def plot_drift_line(df, parameters, color='k', ax=None):

    a, trSteps, v, zStart, si, dx, dt, deadline = convert_params(parameters)
    rt = np.mean(df[df.choice==1].rt.values / dt)
    if v<0:
        rt = np.mean(df[df.choice==0].rt.values / dt)
    accum_x = np.arange(rt)*.001
    driftRate = zStart + (accum_x * v)
    x = np.linspace(trSteps, rt, accum_x.size)
    ax.plot(x, driftRate, color=color, linewidth=3)


def sdt_interact(pH=.80, pFA=.10):

    plt.figure(2)
    ax = plt.gca()

    #n0, n1 = float(FA + CR), float(Hits + Misses)
    n0 = 100; n1 = 100
    if pH == 0:  pH += 0.01
    if pH == n1: pH -= 0.01
    if pFA == 0: pFA += 0.01
    if pFA == n0: pFA -= 0.01

    Hits = pH * n1
    Misses = n1 - Hits
    FA = pFA * n0
    CR = n0 - FA

    d, c = sdt.sdt_mle(Hits, Misses, CR, FA)
    cLine = norm.ppf(1-pFA)
    dstr = "$d'={:.2f}$".format(d)
    cstr = "$c={:.2f}$".format(c)

    x = np.linspace(-4, 7, 1000)
    noiseDist = norm.pdf(loc=0, scale=1, x=x)
    signalDist = norm.pdf(loc=d, scale=1, x=x)

    plt.plot(x, noiseDist, color='k', alpha=.4)
    plt.plot(x, signalDist, color='k')

    yupper = ax.get_ylim()[-1]
    ax.vlines(cLine, 0, yupper, linestyles='-', linewidth=1.5)
    ax.set_ylim(0, yupper)
    ax.set_xlim(-4, 7)
    ax.set_yticklabels([])
    sns.despine(left=True, right=True, top=True)

    ax.text(4, yupper*.9, dstr, fontsize=14)
    ax.text(4, yupper*.8, cstr, fontsize=14)

    plt.show()


def plot_qlearning(data, nblocks=25, analyze=True):

    if analyze:
        auc = utils.get_optimal_auc(data, nblocks, verbose=True)

    sns.set(style='white', font_scale=1.3)
    clrs = ['#3778bf', '#feb308', '#9b59b6', '#2ecc71', '#e74c3c',
            '#3498db', '#fd7f23', '#694098', '#319455', '#f266db',
            '#13579d', '#fa8d67'  '#a38ff1'  '#3caca4', '#c24f54']

    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12,3.5))
    df = data.copy()
    nactions = int(df.columns[-4].split('p')[-1])+1
    actions = np.arange(nactions)

    mudf = df.groupby('trial').mean().reset_index()
    errdf = df.groupby('trial').sem().reset_index()*1.96
    x = mudf.trial.values

    plot_err = True
    if np.isnan(errdf.loc[1, 'q0']):
        plot_err = False

    x3 = np.arange(1, nblocks+1)
    chance = 1/nactions
    mu3, err3 = utils.analyze_bandits(df, nblocks=nblocks, get_err=plot_err)
    ax3.plot(x3, mu3, color='k')
    ax3.hlines(chance, 1, x3[-1], color='k', linestyles='--', label='chance')

    for i, act in enumerate(actions):
        muQ = mudf['q{}'.format(act)].values
        muP = mudf['p{}'.format(act)].values
        ax1.plot(x, muQ, label='$arm_{}$'.format(i), color=clrs[i])
        ax2.plot(x, muP, color=clrs[i])

        if plot_err:
            errQ = errdf['q{}'.format(act)].values
            errP = errdf['p{}'.format(act)].values
            ax1.fill_between(x, muQ-errQ, muQ+errQ, color=clrs[i], alpha=.2)
            ax2.fill_between(x, muP-errP, muP+errP, color=clrs[i], alpha=.2)
            if i==0:
                ax3.fill_between(x3, mu3-err3, mu3+err3, color='k', alpha=.15)
        else:
            ychance = np.ones(mu3.size) * chance
            mu3A = np.copy(mu3)
            mu3B = np.copy(mu3)
            mu3A[np.where(mu3<=chance)] = chance
            mu3B[np.where(mu3>=chance)] = chance
            ax3.fill_between(x3, ychance, mu3A, color='#2ecc71', alpha=.15)
            ax3.fill_between(x3, ychance, mu3B, color='#e74c3c', alpha=.15)

    ax1.legend(loc=4)
    ax1.set_ylabel('$Q(arm)$')
    ax1.set_title('Value')

    ax2.set_ylabel('$P(arm)$')
    ax2.set_ylim(0,1)
    ax2.set_title('Softmax Prob.')

    ax3.set_ylim(0,1)
    ax3.set_ylabel('% Optimal Arm')
    ax3.set_xticks([1, nblocks+1])
    ax3.set_xticklabels([1, df.trial.max()])
    ax3.legend(loc=4)

    for ax in f.axes:
        ax.set_xlabel('Trials')
    plt.tight_layout()
    sns.despine()
