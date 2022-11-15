import torch
from torch import nn

from hive.agents.qnets.mlp import MLPNetwork
from hive.agents.qnets.utils import calculate_output_dim

from  torch.nn.modules.rnn import RNNCellBase

# Where can I move this function?

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

class GRUCell(RNNCellBase):

  # FIXME: Check if there is any signioficance of kwargs in the DV2 config
  def __init__(self, input_size, hidden_size, norm=False, act='Tanh', update_bias=-1): #, **kwargs):
    super().__init__(input_size, hidden_size, bias=norm is not None, num_chunks=3)
    self._act = get_act(act)
    self._norm = norm
    self._update_bias = update_bias
    self._layer = nn.Linear(input_size + hidden_size, 3 * hidden_size, bias=norm is not None) #, **kwargs)
    if norm:
      self._norm = nn.LayerNormalization(dtype=torch.float32)

  @property
  def state_size(self):
    return self._size

  def forward(self, inputs, state):
    state = state[0]  # Keras wraps the state in a list.
    parts = self._layer(torch.cat([inputs, state], -1))
    if self._norm:
      dtype = parts.dtype
      parts = self._norm(parts)
    reset, cand, update = torch.chunk(parts, 3, dim=-1)
    reset = nn.Sigmoid()(reset)
    cand = self._act()(reset * cand)
    update = nn.Sigmoid()(update + self._update_bias)
    output = update * cand + (1 - update) * state
    return output, [output]