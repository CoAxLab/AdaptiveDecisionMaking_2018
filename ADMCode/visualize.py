from __future__ import division
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt


def build_ddm_axis(a, tr, z, tb=1200):

    sns.set(style='white')
    f, ax = plt.subplots(1, figsize=(8.5, 7), sharex=True)
    w = tb
    # tr = tr - 50
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

    plt.setp(axx1, xlim=(xmin - 51, w + 1), ylim=(0 - (.01 * a), a + (.01 * a)))
    plt.setp(axx2, xlim=(xmin - 51, w + 1), ylim=(0 - (.01 * a), a + (.01 * a)))

    axx1.hist([0], normed=False, bins=np.linspace(200, tb, num=9), alpha=1., color='White')
    axx2.hist([0], normed=False, bins=np.linspace(200, tb, num=9), alpha=1., color='White')

    for axx in [axx1, axx2]:
        for spine in ['top', 'left', 'bottom', 'right']:
            axx.spines[spine].set_visible(False)
        axx.set_xticklabels([])
        axx.set_yticklabels([])
    return f, [ax, axx1, axx2]


def plot_ddm_traces(df, traces, parameters, colors=['#3572C6', '#e5344a'], plot_v=False, deadline=1.2):

    a, tr, v, z, si, dx, dt = parameters
    deadline=traces.shape[1]
    zStart = z * a
    trSteps = int(tr/dt)

    rt1 = df[df.choice==1].rt.values / dt
    rt0 = df[df.choice==0].rt.values / dt

    f, axes = build_ddm_axis(a, trSteps, zStart, tb=deadline)
    ax, axx1, axx2 = axes

    clr1, clr0 = colors
    sns.kdeplot(rt1, alpha=.5, linewidth=0, color=clr1, ax=axx1, shade=True,
                clip=(rt1.min(), rt1.max()), bw=35)
    sns.kdeplot(rt0, alpha=.5, linewidth=0, color=clr0, ax=axx2, shade=True,
                clip=(rt0.min(), rt0.max()), bw=35)

    axx1.set_ylim(0, .004)
    axx2.set_ylim(.004, 0.0)
    delay = np.ones(trSteps) * zStart

    for i in range(int(df.shape[0]/2)):
        trace = traces[i]
        c = clr1
        nsteps = np.argmax(trace[trace<=a]) + 2
        if df.iloc[i]['choice']==0:
            c = clr0
            nsteps = np.argmin(trace[trace>=0]) + 2

        y = np.hstack([delay, trace[:nsteps]])
        x = np.arange(delay.size, y.size)
        ax.plot(x, trace[:nsteps], color=c, alpha=.19)

    if plot_v:
        accum_x = np.arange(rt1.mean())*.001
        driftRate = zStart + (accum_x * v)
        x = np.linspace(trSteps, rt1.mean(), accum_x.size)
        ax.plot(x, driftRate, color='k', linewidth=3)
    return ax



def compare_drift_effects(dataframes, param_list, colors=["#3498db", "#f19b2c", '#009e07', '#3572C6', '#e5344a', "#9B59B6"], deadline=1.2):

    a, tr, v, z, si, dx, dt = param_list[0]
    zStart = z * a
    trSteps = int(tr/dt)
    deadline = pd.concat(dataframes).rt.max() / dt

    f, axes = build_ddm_axis(a, trSteps, zStart, tb=deadline)
    ax, axx1, axx2 = axes

    ax.hlines(y=a, xmin=trSteps-100, xmax=deadline, color='k', linewidth=4)
    ax.hlines(y=0, xmin=trSteps-100, xmax=deadline, color='k', linewidth=4)

    for i, df in enumerate(dataframes):

        c = colors[i]
        a, tr, v, z, si, dx, dt = param_list[i]
        zStart = z * a
        trSteps = int(tr/dt)

        rt1 = df[df.choice==1].rt.values / dt - 50
        rt0 = df[df.choice==0].rt.values / dt - 50

        sns.kdeplot(rt1, alpha=.35, linewidth=0, color=c, ax=axx1, shade=True,
                clip=(rt1.min(), rt1.max()), bw=35)
        sns.kdeplot(rt0, alpha=.35, linewidth=0, color=c, ax=axx2, shade=True,
                clip=(rt0.min(), rt0.max()), bw=35)

        accum_x = np.arange(rt1.mean())*.001
        driftRate = zStart + (accum_x * v)
        x = np.linspace(trSteps, rt1.mean(), accum_x.size)
        ax.plot(x, driftRate, color=c, linewidth=3.5)

    axx1.set_ylim(0.0, .004)
    axx2.set_ylim(.004, 0.0)
    return ax
