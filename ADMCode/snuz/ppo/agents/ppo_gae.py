import torch
import numpy as np
from ADMCode.snuz.ppo.utils import log_probability


class Hyperparameters:
    gamma = 0.99
    lam = 0.98
    actor_hidden1 = 64
    actor_hidden2 = 64
    actor_hidden3 = 64
    critic_hidden1 = 64
    critic_lr = 0.0003
    actor_lr = 0.0003
    batch_size = 64
    l2_rate = 0.001
    clip_param = 0.2
    num_training_epochs = 10
    num_episodes = 100
    num_memories = 24
    num_training_epochs = 10
    clip_actions = True
    clip_std = 1.0  #0.25
    seed_value = None


def get_returns(rewards, masks, values, hp):
    rewards = torch.Tensor(rewards)
    masks = torch.Tensor(masks)
    returns = torch.zeros_like(rewards)
    advantages = torch.zeros_like(rewards)

    running_returns = 0
    previous_value = 0
    running_advants = 0

    for t in reversed(range(0, len(rewards))):
        running_returns = rewards[t] + hp.gamma * running_returns * masks[t]
        running_tderror = (
            rewards[t] + hp.gamma * previous_value * masks[t] - values.data[t])
        running_advants = (
            running_tderror + hp.gamma * hp.lam * running_advants * masks[t])

        returns[t] = running_returns
        previous_value = values.data[t]
        advantages[t] = running_advants

    advantages = (advantages - advantages.mean()) / advantages.std()
    return returns, advantages


def surrogate_loss(actor, advantages, states, old_policy, actions, index):
    mu, std, logstd = actor(torch.Tensor(states))
    new_policy = log_probability(actions, mu, std, logstd)
    old_policy = old_policy[index]

    ratio = torch.exp(new_policy - old_policy)
    surrogate = ratio * advantages
    return surrogate, ratio


def train_model(actor,
                critic,
                memory,
                actor_optim,
                critic_optim,
                hp,
                num_training_epochs=10):
    memory = np.array(memory)
    states = np.vstack(memory[:, 0])
    actions = list(memory[:, 1])
    rewards = list(memory[:, 2])
    masks = list(memory[:, 3])
    values = critic(torch.Tensor(states))

    # ----------------------------
    # step 1: get returns and GAEs and log probability of old policy
    returns, advantages = get_returns(rewards, masks, values, hp)
    mu, std, logstd = actor(torch.Tensor(states))
    old_policy = log_probability(torch.Tensor(actions), mu, std, logstd)
    old_values = critic(torch.Tensor(states))

    criterion = torch.nn.MSELoss()
    n = len(states)
    arr = np.arange(n)

    # ----------------------------
    # step 2: get value loss and actor loss and update actor & critic
    for epoch in range(num_training_epochs):
        np.random.shuffle(arr)

        for i in range(n // hp.batch_size):
            batch_index = arr[hp.batch_size * i:hp.batch_size * (i + 1)]
            batch_index = torch.LongTensor(batch_index)
            inputs = torch.Tensor(states)[batch_index]
            returns_samples = returns.unsqueeze(1)[batch_index]
            advantages_samples = advantages.unsqueeze(1)[batch_index]
            actions_samples = torch.Tensor(actions)[batch_index]
            oldvalue_samples = old_values[batch_index].detach()

            loss, ratio = surrogate_loss(actor, advantages_samples, inputs,
                                         old_policy.detach(), actions_samples,
                                         batch_index)

            values = critic(inputs)
            clipped_values = oldvalue_samples + \
                             torch.clamp(values - oldvalue_samples,
                                         -hp.clip_param,
                                         hp.clip_param)
            critic_loss1 = criterion(clipped_values, returns_samples)
            critic_loss2 = criterion(values, returns_samples)
            critic_loss = torch.max(critic_loss1, critic_loss2).mean()

            clipped_ratio = torch.clamp(ratio, 1.0 - hp.clip_param,
                                        1.0 + hp.clip_param)
            clipped_loss = clipped_ratio * advantages_samples
            actor_loss = -torch.min(loss, clipped_loss).mean()

            loss = actor_loss + 0.5 * critic_loss

            critic_optim.zero_grad()
            loss.backward(retain_graph=True)
            critic_optim.step()

            actor_optim.zero_grad()
            loss.backward()
            actor_optim.step()