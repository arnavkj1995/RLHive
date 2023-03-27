import copy 

from json import decoder
import torch
from torch import nn
import torch.distributions as td

from hive.agents.qnets.mlp import MLPNetwork
from hive.agents.qnets.base import FunctionApproximator
from hive.utils.utils import LossFn, OptimizerFn
from hive.utils.schedule import Schedule, PeriodicSchedule

from hive.agents.qnets.utils import (
    InitializationFn,
    calculate_output_dim,
    create_init_weights_fn,
)

class ActorCritic(nn.Module):

  def __init__(self, 
               action_dim,
               feat_dim,
               discrete,
               actor_net: FunctionApproximator,
               critic_net: FunctionApproximator,
               actor_dist: FunctionApproximator,
               critic_dist: FunctionApproximator,
               actor_optimizer_fn: OptimizerFn,
               critic_optimizer_fn: OptimizerFn,
               actor_grad_clip: float = None,
               critic_grad_clip: float = None,
               target_net_soft_update: bool = False,
               target_net_update_fraction: float = 0.05,
               target_net_update_schedule: Schedule = None,
               imag_horizon: int = 15,
               discount_lambda: float = 0.95,
               actor_grad: str = 'dynamics',
               actor_ent: float = 2e-3,
               actor_grad_mix: float = 1.0
              ) -> None:
    super().__init__()

    self._action_dim = action_dim
    self._imag_horizon = imag_horizon
    self._discount_lambda = discount_lambda
    self._actor_grad = actor_grad
    self._actor_ent = actor_ent
    self._actor_grad_mix = actor_grad_mix
    self._actor_grad_clip = actor_grad_clip
    self._critic_grad_clip = critic_grad_clip

    # FIXME: Fix these parameters
    # if self.config.actor.dist == 'auto':
    #   self.config = self.config.update({
    #       'actor.dist': 'onehot' if discrete else 'trunc_normal'})
    
    if actor_grad == 'auto':
      self._actor_grad = 'reinforce' if discrete else 'dynamics'

    actor_network = actor_net(feat_dim)
    actor_dist = actor_dist(self._action_dim, calculate_output_dim(actor_network, feat_dim)[0])
    self.actor = nn.Sequential(actor_network, actor_dist)

    critic_network = critic_net(feat_dim)
    critic_dist = critic_dist(1, calculate_output_dim(critic_network, feat_dim)[0])
    self.critic = nn.Sequential(critic_network, critic_dist)

    self._target_net_soft_update = target_net_soft_update
    if self._target_net_soft_update:
      self._target_critic = copy.deepcopy(self.critic).requires_grad_(False)  
      self._target_net_update_fraction = target_net_update_fraction
      if target_net_update_schedule is None:
        self._target_net_update_schedule = PeriodicSchedule(False, True, 100)
      else:
        self._target_net_update_schedule = target_net_update_schedule()
    else:
      self._target_critic = self.critic

    self.actor_opt = actor_optimizer_fn(self.actor.parameters()) #, **self.config.actor_opt)
    self.critic_opt = critic_optimizer_fn(self.critic.parameters()) #, **self.config.critic_opt)
    # FIXME: Check what this StreamNorm parametere is- Its not getting used for DV2 so skipping it
    # self.rewnorm = common.StreamNorm(**self.config.reward_norm)

  def update(self, world_model, start, is_terminal, reward_fn):
    metrics = {}
    hor = self._imag_horizon
    # The weights are is_terminal flags for the imagination start states.
    # Technically, they should multiply the losses from the second trajectory
    # step onwards, which is the first imagined step. However, we are not
    # training the action that led into the first step anyway, so we can use
    # them to scale the whole sequence.
    # with tf.GradientTape() as actor_tape:
    
    world_model.requires_grad_(False)
    start = {k: v.detach() for k, v in start.items()}
    seq = world_model.imagine(self.actor, start, is_terminal, hor)
    reward = reward_fn(seq)
    seq["reward"] = reward
    mets1 = {}
    # seq['reward'], mets1 = self.rewnorm(reward)
    # mets1 = {f'reward_{k}': v for k, v in mets1.items()}
    target, mets2 = self.target(seq)
    actor_loss, mets3 = self.actor_loss(seq, target)
    
    self.actor_opt.zero_grad()
    actor_loss.backward()
    
    if self._actor_grad_clip is not None:
      torch.nn.utils.clip_grad_value_(
          self.actor.parameters(), self._actor_grad_clip
      )
    
    self.actor_opt.step()
    
    critic_loss, mets4 = self.critic_loss(seq, target)
    self.critic_opt.zero_grad()
    critic_loss.backward()
    if self._critic_grad_clip is not None:
      torch.nn.utils.clip_grad_value_(
          self.critic.parameters(), self._critic_grad_clip
      )
    
    self.critic_opt.step()
    # for param in self.actor.parameters():
    #   print (param.grad)
    
    if self._target_net_update_schedule.update():
      self._update_target()

    # metrics.update(self.actor_opt(actor_tape, actor_loss, self.actor))
    # metrics.update(self.critic_opt(critic_tape, critic_loss, self.critic))
    metrics.update(**mets1, **mets2, **mets3, **mets4)
    world_model.requires_grad_(True)
    return metrics

  def _update_target(self):
    """Update the target network."""
    if self._target_net_soft_update:
      target_params = self._target_critic.state_dict()
      current_params = self.critic.state_dict()
      for key in list(target_params.keys()):
        target_params[key] = (
            1 - self._target_net_update_fraction
        ) * target_params[
            key
        ] + self._target_net_update_fraction * current_params[
            key
        ]
      self._target_critic.load_state_dict(target_params)
    else:
      self._target_critic.load_state_dict(self.critic.state_dict())
        
  def actor_loss(self, seq, target):
    # Actions:      0   [a1]  [a2]   a3
    #                  ^  |  ^  |  ^  |
    #                 /   v /   v /   v
    # States:     [z0]->[z1]-> z2 -> z3
    # Targets:     t0   [t1]  [t2]
    # Baselines:  [v0]  [v1]   v2    v3
    # Entropies:        [e1]  [e2]
    # Weights:    [ 1]  [w1]   w2    w3
    # Loss:              l1    l2
    metrics = {}
    # return self.actor(seq['feat'][:-2]).entropy(), metrics
    # Two states are lost at the end of the trajectory, one for the boostrap
    # value prediction and one because the corresponding action does not lead
    # anywhere anymore. One target is lost at the start of the trajectory
    # because the initial state comes from the replay buffer.
    policy = self.actor(seq['feat'][:-2].detach())

    if self._actor_grad == 'dynamics':
      objective = target[1:]
    elif self.actor_grad == 'reinforce':
      # FIXME- account for Independent dimension
      baseline = self._target_critic(seq['feat'][:-2]) #.mode()
      advantage = target[1:].detach() - baseline.detach()
      action = seq['action'][1:-1].detach()
      objective = policy.log_prob(action) * advantage
    elif self.actor_grad == 'both':
      baseline = self._target_critic(seq['feat'][:-2]) #.mode()
      advantage = target[1:].detach() - baseline.detach()
      objective = policy.log_prob(seq['action'][1:-1]) * advantage
      objective = self._actor_grad_mix * target[1:] + (1 - self._actor_grad_mix) * objective
      metrics['actor_grad_mix'] = self._actor_grad_mix
    else:
      raise NotImplementedError(self._actor_grad)
    ent = policy.entropy() #.sum(axis=-1)

    # print (self._actor_ent.shape, ent.shape)
    # import ipdb
    # ipdb.set_trace()
    objective += self._actor_ent * ent.unsqueeze(-1)
    weight = seq['weight'].detach().unsqueeze(-1)

    actor_loss = -(weight[:-2] * objective).mean()
    metrics['actor_ent'] = ent.mean()
    metrics['actor_ent_scale'] = self._actor_ent
    return actor_loss, metrics

  def critic_loss(self, seq, target):
    # States:     [z0]  [z1]  [z2]   z3
    # Rewards:    [r0]  [r1]  [r2]   r3
    # Values:     [v0]  [v1]  [v2]   v3
    # Weights:    [ 1]  [w1]  [w2]   w3
    # Targets:    [t0]  [t1]  [t2]
    # Loss:        l0    l1    l2
    dist = self.critic(seq['feat'][:-1].detach())
    target = target.detach()
    weight = seq['weight'].detach()
    critic_loss = -(dist.log_prob(target).squeeze(-1) * weight[:-1]).mean()
    metrics = {'critic': critic_loss}
    return critic_loss, metrics

  def lambda_return(
    self, reward, value, pcont, bootstrap, lambda_, axis):
    # Setting lambda=1 gives a discounted Monte Carlo return.
    # Setting lambda=0 gives a fixed 1-step return.
    assert len(reward.shape) == len(value.shape), (reward.shape, value.shape)
    if isinstance(pcont, (int, float)):
      pcont = pcont * torch.ones(reward)
    dims = list(range(len(reward.shape)))
    dims = [axis] + dims[1:axis] + [0] + dims[axis + 1:]
    if axis != 0:
      reward = torch.nn.transpose(reward, dims)
      value = torch.nn.transpose(value, dims)
      pcont = torch.nn.transpose(pcont, dims)
    
    if bootstrap is None:
      bootstrap = torch.zeros(value[-1])
    next_values = torch.cat([value[1:], bootstrap[None]], 0)

    pcont = pcont.unsqueeze(-1)
    inputs = reward + pcont * next_values * (1 - lambda_)
    
    timesteps = list(range(reward.shape[0] - 1, -1, -1))
    
    outputs = []
    accumulated_reward = bootstrap
    for t in timesteps:
        inp = inputs[t]
        discount_factor = pcont[t]
        accumulated_reward = inp + discount_factor * lambda_ * accumulated_reward
        outputs.append(accumulated_reward)

    returns = torch.flip(torch.stack(outputs), [0])

    # if axis != 0:
    #   returns = torch.transpose(returns, dims)
    
    return returns

  def target(self, seq):
    # States:     [z0]  [z1]  [z2]  [z3]
    # Rewards:    [r0]  [r1]  [r2]   r3
    # Values:     [v0]  [v1]  [v2]  [v3]
    # Discount:   [d0]  [d1]  [d2]   d3
    # Targets:     t0    t1    t2
    reward = seq['reward']
    disc = seq['discount']

    value = self._target_critic(seq['feat']).mean

    # Skipping last time step because it is used for bootstrapping.
    target = self.lambda_return(
        reward[:-1], value[:-1], disc[:-1],
        bootstrap=value[-1],
        lambda_=self._discount_lambda,
        axis=0)
    metrics = {}
    metrics['critic_slow'] = value.mean()
    metrics['critic_target'] = target.mean()
    return target, metrics

  # def update_slow_target(self):
  #   if self._slow_target:
  #     if self._updates % self._slow_target_update == 0:
  #       mix = 1.0 if self._updates == 0 else float(
  #           self._slow_target_fraction)
  #       for s, d in zip(self.critic.variables, self._target_critic.variables):
  #         d.assign(mix * s + (1 - mix) * d)
  #     self._updates.assign_add(1)