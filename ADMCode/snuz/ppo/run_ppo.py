"""Test games with flowing actions."""
import os
import errno

from collections import deque

import gym
from gym import wrappers

import numpy as np
import torch
import torch.optim as optim

from ADMCode.snuz.ppo.models import Actor3Sigma
from ADMCode.snuz.ppo.models import Critic3
from ADMCode.snuz.ppo.utils import get_action
from ADMCode.snuz.ppo.utils import save_checkpoint
from ADMCode.snuz.ppo.utils import ZFilter


def run_ppo(env_name='MountainCarContinuous-v0',
            update_every=100,
            save=None,
            progress=True,
            debug=False,
            render=False,
            **algorithm_hyperparameters):

    # ------------------------------------------------------------------------
    from ADMCode.snuz.ppo.agents.ppo_gae import train_model
    from ADMCode.snuz.ppo.agents.ppo_gae import Hyperparameters

    # and its hyperparams
    hp = Hyperparameters()
    for k, v in algorithm_hyperparameters.items():
        setattr(hp, k, v)

    # ------------------------------------------------------------------------
    # Setup the world
    prng = np.random.RandomState(hp.seed_value)

    env = gym.make(env_name)
    env.seed(hp.seed_value)

    num_inputs = env.observation_space.shape[0]
    running_state = ZFilter((num_inputs, ), clip=5)
    num_actions = env.action_space.shape[0]

    # ------------------------------------------------------------------------
    # Actor-critic init
    actor = Actor3Sigma(num_inputs, num_actions, hp, max_std=hp.clip_std)
    critic = Critic3(num_inputs, hp)

    actor_optim = optim.Adam(actor.parameters(), lr=hp.actor_lr)
    critic_optim = optim.Adam(
        critic.parameters(), lr=hp.critic_lr, weight_decay=hp.l2_rate)

    # ------------------------------------------------------------------------
    # Play many games
    episode = 0
    episodes_scores = []
    for n_e in range(hp.num_episodes):
        # Re-init
        actor.eval()
        critic.eval()
        memory = deque()

        # -
        scores = []
        steps = 0
        for n_m in range(hp.num_memories):
            episode += 1
            state = env.reset()
            state = running_state(state)

            score = 0
            done = False
            while not done:
                if render:
                    env.render()

                # Move
                steps += 1
                mu, std, _ = actor(torch.Tensor(state).unsqueeze(0))
                action = get_action(mu, std)[0]  # Flattens too
                action_std = std.clone().detach().numpy().flatten(
                )  # Match action

                if hp.clip_actions:
                    action = np.clip(action, env.action_space.low,
                                     env.action_space.high)

                next_state, reward, done, _ = env.step(action)
                next_state = running_state(next_state)

                # Process outcome
                if done:
                    mask = 0
                else:
                    mask = 1

                # Save/update
                memory.append([state, action, reward, mask, action_std])
                score += reward
                scores.append(score)

                # Shift
                state = next_state

            if debug and (n_m % update_every) == 0:
                print(">>> Mem. {}".format(n_m))
                print(">>> Last score {}".format(score))
                print(">>> Mu, Sigma ({}, {})".format(mu.tolist(),
                                                      std.tolist()))

        score_avg = np.mean(scores)
        if progress:
            print(">>> Episode {} avg. score {}".format(n_e, score_avg))
        episodes_scores.append(score_avg)

        # --------------------------------------------------------------------
        # Learn!
        actor.train()
        critic.train()
        train_model(
            actor,
            critic,
            memory,
            actor_optim,
            critic_optim,
            hp,
            num_training_epochs=hp.num_training_epochs)

        # --------------------------------------------------------------------
        if (save is not None) and (n_e % update_every) == 0:
            save_checkpoint({
                'actor': actor.state_dict(),
                'critic': critic.state_dict(),
                'z_filter_n': running_state.rs.n,
                'z_filter_m': running_state.rs.mean,
                'z_filter_s': running_state.rs.sum_square,
                'score': score_avg
            },
                            filename=save + "_ep_{}.pytorch.tar".format(n_e))

    return list(range(hp.num_episodes)), episodes_scores
