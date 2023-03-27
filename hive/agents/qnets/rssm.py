import torch
from torch import nn
import torch.distributions as td

from hive.agents.qnets.base import FunctionApproximator
from hive.agents.qnets.rnn import GRUCell
# from hive.agents.qnets.gru import GRUCell

# TODO: Add the Cell layer (GRU with norm layer)
# TODO: Add the For loop for observe and imagine functions
class RSSM(nn.Module):

  # FIXME: Find the substitute of Mixed precision policy in Pytorch
  def __init__(
      self, num_actions, device, stoch=30, deter=200, hidden=200, 
      embed=1024, discrete=False, act='elu', norm='none', 
      std_act='sigmoid2', min_std=0.1):
    super().__init__()
    self._device = torch.device("cpu" if not torch.cuda.is_available() else device)

    self._stoch = stoch
    self._deter = deter
    self._hidden = hidden
    self._discrete = discrete
    self._embed = embed
    self._num_actions = num_actions
    self._act = get_act(act)
    self._norm = norm
    self._std_act = std_act
    self._min_std = min_std
    # FIXME: Add the GRUCell with LayerNorm later
    self._cell = GRUCell(self._deter, self._hidden)
    self._prior = self._build_stochastic_layer(type='prior')
    self._posterior = self._build_stochastic_layer(type='posterior')
    self._embed_state_action = self._build_embed_state_action() 
    # self._cast = lambda x: tf.cast(x, prec.global_policy().compute_dtype)

  def _build_embed_state_action(self):
      """
      model is supposed to take in previous stochastic state and previous action 
      and embed it to deter size for rnn input
      """
      # FIXME: get substitute for action size
      if self._discrete:
        layers = [nn.Linear(self._stoch * self._discrete + self._num_actions, self._deter)]
      else:
        layers = [nn.Linear(self._stoch + self._num_actions, self._deter)]
      if self._norm == 'layer':
        layers += [nn.LayerNorm(self._hidden)]
      layers += [self._act()]
      return nn.Sequential(*layers)

  def _build_stochastic_layer(self, type='prior'):
      """
      model is supposed to take in latest deterministic state 
      and output prior over stochastic state
      """
      if type == 'prior':
        layers = [nn.Linear(self._deter, self._hidden)]
      else:
        layers = [nn.Linear(self._deter + self._embed, self._hidden)]
      if self._norm == 'layer':
        layers += [nn.LayerNorm(self._hidden)]
      layers += [self._act()]
      if self._discrete:
        layers += [nn.Linear(self._hidden, self._stoch * self._discrete)]
      else:
        layers += [nn.Linear(self._hidden, 2 * self._stoch)]

      return nn.Sequential(*layers)

  def _unflatten(self, input):
    return {k: torch.stack([x[k] for x in input]) for k in input[0]}

  def initial(self, batch_size):
    # dtype = prec.global_policy().compute_dtype
    if self._discrete:
      state = dict(
          logit=torch.zeros([batch_size, self._stoch, self._discrete]).to(self._device),
          stoch=torch.zeros([batch_size, self._stoch, self._discrete]).to(self._device),
          deter=torch.zeros([batch_size, self._deter]).to(self._device))
    else:
      state = dict(
          mean=torch.zeros([batch_size, self._stoch]).to(self._device),
          std=torch.zeros([batch_size, self._stoch]).to(self._device),
          stoch=torch.zeros([batch_size, self._stoch]).to(self._device),
          deter=torch.zeros([batch_size, self._deter]).to(self._device))

    return state

  # FIXME: Add the 'For' loop for observe and imagine functions
  def observe(self, embed, action, is_first, state=None):
    swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
    if state is None:
      state = self.initial(action.size(dim=0))

    # print (action.shape,)
    action = swap(action)
    embed = swap(embed)
    is_first = swap(is_first)

    # FIXME: FInd substiture for static_scan in common
    horizon = action.size(dim=0)
    posts, priors = [], []
    for t in range(horizon):
      post, prior = self.obs_step(state, action[t], embed[t], is_first[t])
      posts.append(post)
      priors.append(prior)
      state = post

    posts = self._unflatten(posts)
    priors = self._unflatten(priors)

    posts = {k: swap(v) for k, v in posts.items()}
    priors = {k: swap(v) for k, v in priors.items()}

      # action_entropy.append(action_dist.entropy())
      # imag_log_probs.append(action_dist.log_prob(torch.round(action.detach())))

    # post, prior = common.static_scan(
    #     lambda prev, inputs: self.obs_step(prev[0], *inputs),
    #     (swap(action), swap(embed), swap(is_first)), (state, state))
    return posts, priors

  # @tf.function
  def imagine(self, action, state=None):
    swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
    if state is None:
      state = self.initial(action.size(dim=0))

    assert isinstance(state, dict), state
    action = swap(action)

    horizon = action.size(dim=0)

    # FIXME: FInd substiture for static_scan in common
    priors = []
    for t in range(horizon):
      prior = self.img_step(state, action[t])
      priors.append(prior)
      state = prior

    priors = self._unflatten(priors)
    priors = {k: swap(v) for k, v in priors.items()}

    return priors
  
  def get_feat(self, state):
    stoch = state['stoch']
    if self._discrete:
      shape = list(stoch.shape[:-2]) + [self._stoch * self._discrete]
      stoch = stoch.view(shape)
    return torch.cat([state['deter'], stoch], -1)

  def get_dist(self, state):
    if self._discrete:
        return td.OneHotCategoricalStraightThrough(logits=state["logit"])
    else:
        return td.Normal(state["mean"], state["std"])

  # @tf.function
  # TODO: Check what the is_first flag is doing
  def obs_step(self, prev_state, prev_action, embed, is_first, sample=True):
    # if is_first.any():
    # TODO: Need to understand the line below
    # prev_state, prev_action = tf.nest.map_structure(
    #     lambda x: tf.einsum(
    #         'b,b...->b...', 1.0 - is_first.astype(x.dtype), x),
    #     (prev_state, prev_action))
    prior = self.img_step(prev_state, prev_action, sample)
    x = torch.cat([prior['deter'], embed], -1)
    stats = self._posterior(x)
    stats = self._suff_stats_layer(stats)
    dist = self.get_dist(stats)
    stoch = dist.rsample() if sample else dist.mode()
    post = {'stoch': stoch, 'deter': prior['deter'], **stats}
    return post, prior

  # @tf.function
  def img_step(self, prev_state, prev_action, sample=True):
    # prev_stoch = self._cast(prev_state['stoch'])
    # prev_action = self._cast(prev_action)
    prev_stoch = prev_state['stoch']
    if self._discrete:
      shape = list(prev_stoch.shape[:-2]) + [self._stoch * self._discrete]
      prev_stoch = torch.reshape(prev_stoch, shape)
    x = torch.cat([prev_stoch, prev_action], -1)
    # print (x.shape)
    x = self._embed_state_action(x)
    deter = prev_state['deter']
    # FIXME: remove the GRU, worst hack till date :P
    # x = deter
    x, deter = self._cell(x, [deter])
    deter = deter[0]  # Keras wraps the state in a list.
    
    stats = self._prior(x)      
    stats = self._suff_stats_layer(stats)

    dist = self.get_dist(stats)
    stoch = dist.rsample() if sample else dist.mode()
    prior = {'stoch': stoch, 'deter': deter, **stats}
    return prior

  def _suff_stats_layer(self, x):
    if self._discrete:
      logit = x.view(list(x.shape[:-1]) + [self._stoch, self._discrete])
      return {'logit': logit}
    else:
      mean, std = torch.chunk(x, 2, dim=-1)
      std = {
          'softplus': lambda: nn.functional.softplus(std),
          'sigmoid': lambda: nn.functional.sigmoid(std),
          'sigmoid2': lambda: 2 * nn.functional.sigmoid(std / 2),
      }[self._std_act]()
      std = std + self._min_std
      return {'mean': mean, 'std': std}

  # TODO: Can be done in the end
  def kl_loss(self, post, prior, forward, balance, free, free_avg):
    kld = td.kl.kl_divergence
    # sg = lambda x: tf.nest.map_structure(tf.stop_gradient, x)
    lhs, rhs = (prior, post) if forward else (post, prior)
    mix = balance if forward else (1 - balance)
    # FIXME: Adding .sum(axis=-1) to account for independedent axis
    if balance == 0.5:
      value = kld(self.get_dist(lhs), self.get_dist(rhs)).sum(axis=-1)
      loss = torch.max(value, free).mean()
    else:
      value_lhs = value = kld(self.get_dist(lhs), self.get_dist({x: rhs[x].detach() for x in rhs})).sum(axis=-1)
      value_rhs = kld(self.get_dist({x: lhs[x].detach() for x in lhs}), self.get_dist(rhs)).sum(axis=-1)
      if free_avg:
        loss_lhs = torch.clamp(value_lhs.mean(), min=torch.tensor(free))
        loss_rhs = torch.clamp(value_rhs.mean(), min=torch.tensor(free))
      else:
        # FIXME: Check if the CLIP function is differentiable
        loss_lhs = torch.clamp(value_lhs, min=free).mean()
        loss_rhs = torch.clamp(value_rhs, min=free).mean()
      loss = mix * loss_lhs + (1 - mix) * loss_rhs
    return loss, value

def get_act(name):
  if name == 'none':
    return nn.identity
  if name == 'mish':
    return lambda x: x * nn.Tanh(nn.Softplus(x))
  elif hasattr(nn, name):
    return getattr(nn, name)
  elif hasattr(nn, name):
    return getattr(nn, name)
  else:
    raise NotImplementedError(name)