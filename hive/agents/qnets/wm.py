from json import decoder
import torch
from torch import nn
import torch.distributions as td

from hive.agents.qnets.base import FunctionApproximator
from hive.utils.utils import LossFn, OptimizerFn

from hive.agents.qnets.utils import (
    InitializationFn,
    calculate_output_dim,
    create_init_weights_fn,
)

# TODO: Add the Cell layer (GRU with norm layer)
# TODO: Add the For loop for observe and imagine functions
class WorldModel(nn.Module):

  # FIXME: Check if tfstep is required
  def __init__(self, 
              obs_space,
              action_dim,
              device="cpu",
              grad_heads = None,
              discount: float = 0.99,
              loss_scales: dict = None,
              grad_clip: float = None,
              clip_rewards: str = 'identity',
              transition_net: FunctionApproximator = None, 
              encoder_net: FunctionApproximator = None, 
              decoder_net: FunctionApproximator = None,
              decoder_dist: FunctionApproximator = None, 
              reward_net: FunctionApproximator = None, 
              reward_dist: FunctionApproximator = None,
              discount_net: FunctionApproximator = None,
              discount_dist: FunctionApproximator = None,
              model_optimizer_fn: OptimizerFn = None):

    super().__init__()

    self.obs_space = obs_space
    self._device = torch.device("cpu" if not torch.cuda.is_available() else device)
    self._grad_heads = grad_heads
    self._discount = discount
    self._clip_rewards = clip_rewards
    self._grad_clip = grad_clip
    self._loss_scales = loss_scales or {}

    # discrete = hasattr(self._act_space, 'n')
    # action_dim = self._act_space.n if discrete else self._act_space.shape[0]

    # shapes = {'image': obs_space.shape, 'action': self._act_space.n, 'reward': 1, 'discount': 1}
    
    shapes = {'image': obs_space.shape, 'action': action_dim, 'reward': 1, 'discount': 1}
    # shapes = {k: tuple(v.shape) for k, v in obs_space.items()}
    self.encoder = encoder_net(shapes["image"])
    
    # FIXME: calculate embed here
    self.rssm = transition_net(action_dim, self._device)
    
    # FIXME: Pass the right parameters and update the ConvNetwork with Norm layers
    self.heads = {}

    # Add the deconv network
    state_dim = self.rssm._deter + self.rssm._stoch
    decoder_network = decoder_net(state_dim)
    decoder_dist_fn = decoder_dist(shapes['image'], None)
    self.heads['observation'] = nn.Sequential(decoder_network, decoder_dist_fn).to(self._device)

    reward_network = reward_net(state_dim)
    reward_dist_fn = reward_dist(1, calculate_output_dim(reward_network, state_dim)[0])
    self.heads['reward'] = nn.Sequential(reward_network, reward_dist_fn).to(self._device)

    if discount_net:
      discount_network = discount_net(state_dim)
      discount_dist_fn = discount_dist(1, calculate_output_dim(discount_network, state_dim)[0])
      self.heads['discount'] = nn.Sequential(discount_network, discount_dist_fn).to(self._device)
      
    for name in grad_heads:
      assert name in self.heads, name
    
    # Look at the optimizer stuff later
    # if model_optimizer_fn is None:
    #   model_optimizer_fn = torch.optim.Adam

    self._model_parameters = list(self.heads['reward'].parameters()) + list(self.heads['observation'].parameters()) + list(self.encoder.parameters()) + list(self.rssm.parameters()) #+ [x.parameters() for x in self.heads.values()]
    self._model_optimizer = model_optimizer_fn(self._model_parameters)

  def update(self, data, state=None):
    # with tf.GradientTape() as model_tape:
    model_loss, state, outputs, metrics = self.loss(data, state)
    self._model_optimizer.zero_grad()
    model_loss.backward()
    if self._grad_clip is not None:
      torch.nn.utils.clip_grad_value_(
          self._model_parameters, self._grad_clip
      )
    
    self._model_optimizer.step()
    # modules = [self.encoder, self.rssm, *self.heads.values()]
    # metrics.update(self.model_opt(model_tape, model_loss, modules))
    return state, outputs, metrics

  def loss(self, data, state=None):
    # TODO: Check the preprocess later 
    data = self.preprocess(data)
    obs = data['observation'].view(-1, *data['observation'].shape[2:])

    embed = self.encoder(obs)
    embed = embed.view(data['observation'].shape[0], data['observation'].shape[1], -1)

    post, prior = self.rssm.observe(
        embed, data['action'], data['is_first'], state)
    
    # FIXME: Add the config params for the KL Loss
    kl_loss, kl_value = self.rssm.kl_loss(post, prior, forward=False, balance=0.8, free=0.00001, free_avg=False) # **self.config.kl)
    assert len(kl_loss.shape) == 0
    likes = {}
    losses = {'kl': kl_loss}
    feat = self.rssm.get_feat(post)
    
    for name, head in self.heads.items():
      grad_head = (name in self._grad_heads)
      inp = feat if grad_head else feat.detach()

      out = head(inp)
      
      dists = out if isinstance(out, dict) else {name: out}

      for key, dist in dists.items():
        like = dist.log_prob(data[key])
        likes[key] = like
        losses[key] = -like.mean()

    model_loss = sum(
        self._loss_scales.get(k, 1.0) * v for k, v in losses.items())

    outs = dict(
        embed=embed, feat=feat, post=post,
        prior=prior, likes=likes, kl=kl_value)

    metrics = {f'{name}_loss': value for name, value in losses.items()}
    metrics['model_kl'] = kl_value.mean()
    metrics['prior_ent'] = self.rssm.get_dist(prior).entropy().mean()
    metrics['post_ent'] = self.rssm.get_dist(post).entropy().mean()
    last_state = {k: v[:, -1] for k, v in post.items()}
    return model_loss, last_state, outs, metrics

  def imagine(self, policy, start, is_terminal, horizon):
    flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
    start = {k: flatten(v) for k, v in start.items()}
    start['feat'] = self.rssm.get_feat(start)
    # print (" After get feat function ", policy(start['feat']).mode.shape)
    start['action'] = torch.zeros_like(policy(start['feat']).mode)
    seq = {k: [v] for k, v in start.items()}
    for _ in range(horizon):
      action = policy(seq['feat'][-1].detach()).rsample()
      state = self.rssm.img_step({k: v[-1] for k, v in seq.items()}, action)
      feat = self.rssm.get_feat(state)
      for key, value in {**state, 'action': action, 'feat': feat}.items():
        seq[key].append(value)
    seq = {k: torch.stack(v, 0) for k, v in seq.items()}
    if 'discount' in self.heads:
      disc = self.heads['discount'](seq['feat']).mean
      if is_terminal is not None:
        # Override discount prediction for the first step with the true
        # discount factor from the replay buffer.
        true_first = 1.0 - flatten(is_terminal).astype(disc.dtype)
        true_first *= self.config.discount
        disc = torch.cat([true_first[None], disc[1:]], 0)
    else:
      disc = self._discount * torch.ones(seq['feat'].shape[:-1]).to(self._device)
    seq['discount'] = disc
    # Shift discount factors because they imply whether the following state
    # will be valid, not whether the current state is valid.
    seq['weight'] = torch.cumprod(
        torch.cat([torch.ones_like(disc[:1]), disc[:-1]], 0), 0)
    return seq

#   @tf.function
  def preprocess(self, obs):
    dtype = torch.float16
    # dtype = prec.global_policy().compute_dtype
    # FIXME: Why is this copy here? Is this required?

    # obs = obs.copy()
    for key, value in obs.items():
      if key.startswith('log_'):
        continue
      if value.dtype == torch.int32:
        value = value.type(dtype)
      # FIXME: wierd hardcoding for uint8 :P
      if value.dtype == torch.uint8:
        value = value.type(dtype) / 255.0 - 0.5
      obs[key] = value

    return obs

#   @tf.function
  def video_pred(self, data, key):
    decoder = self.heads['decoder']
    truth = data[key][:6] + 0.5
    embed = self.encoder(data)
    states, _ = self.rssm.observe(
        embed[:6, :5], data['action'][:6, :5], data['is_first'][:6, :5])
    recon = decoder(self.rssm.get_feat(states))[key].mode()[:6]
    init = {k: v[:, -1] for k, v in states.items()}
    prior = self.rssm.imagine(data['action'][:6, 5:], init)
    openl = decoder(self.rssm.get_feat(prior))[key].mode()
    model = torch.concat([recon[:, :5] + 0.5, openl + 0.5], 1)
    error = (model - truth + 1) / 2
    video = torch.concat([truth, model, error], 2)
    B, T, H, W, C = video.shape
    return video.transpose((1, 2, 0, 3, 4)).reshape((T, H, B * W, C))
