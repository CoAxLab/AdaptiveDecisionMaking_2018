#!/usr/local/bin/env python
from __future__ import division
import numpy as np
from numpy import array
from numpy.random import sample as rs
from numpy import newaxis as na
import pandas as pd
from scipy.stats import sem
import seaborn as sns
import string
import matplotlib.pyplot as plt



def update_Qi(Qval, reward, alpha):
    """ update q-value of selected action, given reward and alpha
    """
    return Qval + alpha * (reward - Qval)


def update_Pall(Qvector, beta):
    """ update vector of action selection probabilities given
    associated q-values
    """
    return np.array([np.exp(beta*Q_i) / np.sum(np.exp(beta * Qvector)) for Q_i in Qvector])




class MultiArmedBandit(object):
    """ defines a multi-armed bandit task

    ::Arguments::
        preward (list): 1xN vector of reward probaiblities for each of N bandits
        rvalues (list): 1xN vector of payout values for each of N bandits
    """
    def __init__(self, preward=[.9, .8, .7], rvalues=[1, 1, 1]):
        self.preward = preward
        self.rvalues = rvalues
        try:
            assert(len(self.rvalues)==len(self.preward))
        except AssertionError:
            self.rvalues = np.ones(len(self.preward))

    def set_params(self, **kwargs):
        error_msg = """preward and rvalues must be same size
                    setting all rvalues to 1"""
        kw_keys = list(kwargs)
        if 'preward' in kw_keys:
            self.preward = kwargs['preward']
            if 'rvalues' not in kw_keys:
                try:
                    assert(len(self.rvalues)==len(self.preward))
                except AssertionError:
                    self.rvalues = np.ones(len(self.preward))

        if 'rvalues' in kw_keys:
            self.rvalues = kwargs['rvalues']
            try:
                assert(len(self.rvalues)==len(self.preward))
            except AssertionError:
                raise(AssertionError, error_msg)


    def get_feedback(self, action_ix):
        pOutcomes = np.array([self.preward[action_ix], 1-self.preward[action_ix]])
        Outcomes = np.array([self.rvalues[action_ix], 0])
        feedback = np.random.choice(Outcomes, p=pOutcomes)
        return feedback




class Qagent(object):
    """ defines the learning parameters of single q-learning agent
    in a multi-armed bandit task

    ::Arguments::
        alpha (float): learning rate
        beta (float): inverse temperature parameter
        preward (list): 1xN vector of reward probaiblities for each of N bandits
        rvalues (list): 1xN vector of payout values for each of N bandits
                        IF rvalues is None, all values set to 1

    """
    def __init__(self, alpha=.04, beta=3.5, epsilon=.1, preward=[.9, .8, .7], rvalues=None):
        if rvalues is None:
            rvalues = np.ones(len(preward))
        self.bandits = MultiArmedBandit(preward=preward, rvalues=rvalues)
        self.updateQ = lambda Qval, r, alpha: Qval + alpha*(r - Qval)
        self.updateP = lambda Qvector, act_i, beta: np.exp(beta*Qvector[act_i])/np.sum(np.exp(beta*Qvector))
        self.set_params(alpha=alpha, beta=beta, epsilon=epsilon)


    def set_params(self, **kwargs):
        """ update learning rate, inv. temperature, and/or
        epsilon parameters of q-learning agent
        """

        kw_keys = list(kwargs)

        if 'alpha' in kw_keys:
            self.alpha = kwargs['alpha']

        if 'beta' in kw_keys:
            self.beta = kwargs['beta']

        if 'epsilon' in kw_keys:
            self.epsilon = kwargs['epsilon']

        if 'preward' in kw_keys:
            self.bandits.set_params(preward=kwargs['preward'])

        if 'rvalues' in kw_keys:
            self.bandits.set_params(rvalues=kwargs['rvalues'])

        self.nact = len(self.bandits.preward)
        self.actions = np.arange(self.nact)


    def play_bandits(self, ntrials=1000, get_output=True):
        """ simulates agent performance on a multi-armed bandit task

        ::Arguments::
            ntrials (int): number of trials to play bandits
            get_output (bool): returns output DF if True (default)

        ::Returns::
            DataFrame (Ntrials x Nbandits) with trialwise Q and P
            values for each bandit
        """
        pdata = np.zeros((ntrials+1, self.nact))
        pdata[0, :] = np.array([1/self.nact]*self.nact)
        qdata = np.zeros_like(pdata)
        self.choices = []
        self.feedback = []

        for t in range(ntrials):

            # select bandit arm (action)
            act_i = np.random.choice(self.actions, p=pdata[t, :])

            # observe feedback
            r = self.bandits.get_feedback(act_i)

            # update value of selected action
            qdata[t+1, act_i] = update_Qi(qdata[t, act_i], r, self.alpha)

            # broadcast old q-values for unchosen actions
            for act_j in self.actions[np.where(self.actions!=act_i)]:
                qdata[t+1, act_j] = qdata[t, act_j]

            # update action selection probabilities and store data
            pdata[t+1, :] = update_Pall(qdata[t+1, :], self.beta)
            self.choices.append(act_i)
            self.feedback.append(r)

        self.pdata = pdata[1:, :]
        self.qdata = qdata[1:, :]
        self.make_output_df()

        if get_output:
            return self.data.copy()


    def make_output_df(self):
        """ generate output dataframe with trialwise Q and P measures for each bandit,
        as well as choice selection, and feedback
        """
        df = pd.concat([pd.DataFrame(dat) for dat in [self.qdata, self.pdata]], axis=1)
        columns = np.hstack(([['{}{}'.format(x, c) for c in self.actions] for x in ['q', 'p']]))
        df.columns = columns
        df.insert(0, 'trial', np.arange(1, df.shape[0]+1))
        df['choice'] = self.choices
        df['feedback'] = self.feedback
        r = np.array(self.bandits.rvalues)
        p = np.array(self.bandits.preward)
        df['optimal'] = np.where(df['choice']==np.argmax(p * r), 1, 0)
        df.insert(0, 'agent', 1)
        self.data = df.copy()


    def simulate_multiple(self, nsims=10, ntrials=1000):
        """ simulates multiple identical agents on multi-armed bandit task
        """
        dflist = []
        for i in range(nsims):
            data_i = self.play_bandits(ntrials=ntrials, get_output=True)
            data_i['agent'] += i
            dflist.append(data_i)
        return pd.concat(dflist)
