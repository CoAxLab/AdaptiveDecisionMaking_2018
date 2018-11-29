import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------------
# Code from
# https://github.com/reinforcement-learning-kr/pg_travel/blob/master/unity/model.py


class Actor3Linear(nn.Module):
    """Three layer MLP."""

    def __init__(self, num_inputs, num_outputs, hp, max_std=1):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.max_std = max_std

        super(Actor3Linear, self).__init__()

        # Latent
        self.fc1 = nn.Linear(num_inputs, hp.actor_hidden1)
        self.fc2 = nn.Linear(hp.actor_hidden1, hp.actor_hidden2)

        # Note: fc3 and fc4 are in parallel!
        # Mu
        self.fc3 = nn.Linear(hp.actor_hidden2, num_outputs)
        self.fc3.weight.data.mul_(0.1)
        self.fc3.bias.data.mul_(0.0)
        # Sigma
        self.fc4 = nn.Linear(hp.actor_hidden2, num_outputs)
        self.fc4.weight.data.mul_(0.1)
        self.fc4.bias.data.mul_(0.0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        mu = self.fc3(x)

        std = torch.exp(self.fc4(x))
        std = torch.clamp(std, 0, self.max_std)
        logstd = torch.log(std)

        return mu, std, logstd


class Actor3Sigma(nn.Module):
    """Three layer MLP."""

    def __init__(self, num_inputs, num_outputs, hp, max_std=1):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.max_std = max_std

        super(Actor3Sigma, self).__init__()

        # Latent
        self.fc1 = nn.Linear(num_inputs, hp.actor_hidden1)
        self.fc2 = nn.Linear(hp.actor_hidden1, hp.actor_hidden2)

        # Note: fc3 and fc4 are in parallel!
        # Mu
        self.fc3 = nn.Linear(hp.actor_hidden2, num_outputs)
        self.fc3.weight.data.mul_(0.1)
        self.fc3.bias.data.mul_(0.0)
        # Sigma
        self.fc4 = nn.Linear(hp.actor_hidden2, num_outputs)
        self.fc4.weight.data.mul_(0.1)
        self.fc4.bias.data.mul_(0.0)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        mu = self.fc3(x)

        std = torch.exp(self.fc4(x))
        std = torch.clamp(std, 0, self.max_std)
        logstd = torch.log(std)

        return mu, std, logstd


class Actor3(nn.Module):
    """Three layer MLP."""

    def __init__(self, num_inputs, num_outputs, hp):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        super(Actor3, self).__init__()

        self.fc1 = nn.Linear(num_inputs, hp.actor_hidden1)
        self.fc2 = nn.Linear(hp.actor_hidden1, hp.actor_hidden2)
        self.fc3 = nn.Linear(hp.actor_hidden2, num_outputs)
        self.fc3.weight.data.mul_(0.1)
        self.fc3.bias.data.mul_(0.0)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        mu = self.fc3(x)
        logstd = torch.zeros_like(mu)
        std = torch.exp(logstd)
        return mu, std, logstd


class Actor4(nn.Module):
    """Four layer MLP."""

    def __init__(self, num_inputs, num_outputs, hp):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        super(Actor4, self).__init__()
        self.fc1 = nn.Linear(num_inputs, hp.actor_hidden1)
        self.fc2 = nn.Linear(hp.actor_hidden1, hp.actor_hidden2)
        self.fc3 = nn.Linear(hp.actor_hidden2, hp.actor_hidden3)
        self.fc4 = nn.Linear(hp.actor_hidden4, num_outputs)

        self.fc4.weight.data.mul_(0.1)
        self.fc4.bias.data.mul_(0.0)

    def forward(self, x):
        if self.args.activation == 'tanh':
            x = F.tanh(self.fc1(x))
            x = F.tanh(self.fc2(x))
            x = F.tanh(self.fc3(x))
            mu = self.fc4(x)
        elif self.args.activation == 'swish':
            x = self.fc1(x)
            x = x * F.sigmoid(x)
            x = self.fc2(x)
            x = x * F.sigmoid(x)
            x = self.fc3(x)
            x = x * F.sigmoid(x)
            mu = self.fc4(x)
        else:
            raise ValueError

        logstd = torch.zeros_like(mu)
        std = torch.exp(logstd)
        return mu, std, logstd


class Critic3Linear(nn.Module):
    def __init__(self, num_inputs, hp):
        super(Critic3Linear, self).__init__()
        self.fc1 = nn.Linear(num_inputs, hp.critic_hidden1)
        self.fc2 = nn.Linear(hp.critic_hidden1, hp.critic_hidden1)
        self.fc3 = nn.Linear(hp.critic_hidden1, 1)
        self.fc3.weight.data.mul_(0.1)
        self.fc3.bias.data.mul_(0.0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        v = self.fc3(x)
        return v


class Critic3(nn.Module):
    def __init__(self, num_inputs, hp):
        super(Critic3, self).__init__()
        self.fc1 = nn.Linear(num_inputs, hp.critic_hidden1)
        self.fc2 = nn.Linear(hp.critic_hidden1, hp.critic_hidden1)
        self.fc3 = nn.Linear(hp.critic_hidden1, 1)
        self.fc3.weight.data.mul_(0.1)
        self.fc3.bias.data.mul_(0.0)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        v = self.fc3(x)
        return v


# ----------------------------------------------------------------------------
# Other models (experimental)
class ActorSigma1(nn.Module):
    """A N(mu, sigma) parameterized policy model.
    
    Note: sigma is learnable; this implementation is shallow."""

    def __init__(self,
                 in_channels,
                 action_space,
                 num_hidden1=128,
                 gain=1,
                 sigma=None):
        super(ActorSigma1, self).__init__()
        self.gain = gain

        # Est sigma?
        if sigma is not None:
            self.sigma0 = torch.tensor(sigma)
        else:
            self.sigma0 = sigma

        # Def number of outputs, per param (mu, sigma)
        num_outputs = action_space.shape[0]
        self.action_space = action_space

        # Def the network
        # Shared intial
        self.fc1 = nn.Linear(in_channels, num_hidden1)

        # Mu
        self.fc_mu = nn.Linear(num_hidden1, num_outputs)
        self.fc_mu.bias.data.zero_()

        # Sigma?
        if self.sigma0 is None:
            self.fc_sigma = nn.Linear(num_hidden1, num_outputs)
            self.fc_sigma.bias.data.zero_()

    def forward(self, x):
        # Shared nonlin. projection
        x = F.softmax(self.fc1(x))

        # Linear mu
        mu = self.fc_mu(x)

        # Exp. sigma
        if self.sigma0 is None:
            sigma = torch.exp(self.fc_sigma(x) - self.gain)
            # print(sigma)
        else:
            sigma = self.sigma0

        return mu, sigma


class DiscreteMLPPolicy(nn.Module):
    """A discrete-action policy model."""

    def __init__(self, in_channels, num_action=2, num_hidden1=128):
        super(DiscreteMLPPolicy, self).__init__()
        self.affine1 = nn.Linear(in_channels, num_hidden1)
        self.affine2 = nn.Linear(num_hidden1, num_action)

    def forward(self, x):
        x = F.relu(self.affine1(x))
