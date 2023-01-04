import math
import numpy as np 
from numbers import Number

import torch
import torch.nn as nn
import torch.distributions as td
from torch.distributions import Distribution, constraints
from torch.distributions.utils import broadcast_all

CONST_SQRT_2 = math.sqrt(2)
CONST_INV_SQRT_2PI = 1 / math.sqrt(2 * math.pi)
CONST_INV_SQRT_2 = 1 / math.sqrt(2)
CONST_LOG_INV_SQRT_2PI = math.log(CONST_INV_SQRT_2PI)
CONST_LOG_SQRT_2PI_E = 0.5 * math.log(2 * math.pi * math.e)


class TruncatedStandardNormal(Distribution):
    """
    Truncated Standard Normal distribution
    https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    """

    arg_constraints = {
        'a': constraints.real,
        'b': constraints.real,
    }
    has_rsample = True

    def __init__(self, a, b, validate_args=None):
        self.a, self.b = broadcast_all(a, b)
        if isinstance(a, Number) and isinstance(b, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.a.size()
        super(TruncatedStandardNormal, self).__init__(batch_shape, validate_args=validate_args)
        if self.a.dtype != self.b.dtype:
            raise ValueError('Truncation bounds types are different')
        if any((self.a >= self.b).view(-1, ).tolist()):
            raise ValueError('Incorrect truncation range')
        eps = torch.finfo(self.a.dtype).eps
        self._dtype_min_gt_0 = eps
        self._dtype_max_lt_1 = 1 - eps
        self._little_phi_a = self._little_phi(self.a)
        self._little_phi_b = self._little_phi(self.b)
        self._big_phi_a = self._big_phi(self.a)
        self._big_phi_b = self._big_phi(self.b)
        self._Z = (self._big_phi_b - self._big_phi_a).clamp_min(eps)
        self._log_Z = self._Z.log()
        self._lpbb_m_lpaa_d_Z = (self._little_phi_b * self.b - self._little_phi_a * self.a) / self._Z
        self._mean = -(self._little_phi_b - self._little_phi_a) / self._Z
        self._mode = torch.minimum(torch.maximum(torch.ones_like(self.a), self.a), self.b)  # TODO pull request?
        self._variance = 1 - self._lpbb_m_lpaa_d_Z - ((self._little_phi_b - self._little_phi_a) / self._Z) ** 2
        self._entropy = CONST_LOG_SQRT_2PI_E + self._log_Z - 0.5 * self._lpbb_m_lpaa_d_Z

    @constraints.dependent_property
    def support(self):
        return constraints.interval(self.a, self.b)

    @property
    def mean(self):
        return self._mean

    @property
    def mode(self):
        return self._mode

    @property
    def variance(self):
        return self._variance

    # @property #In pytorch is a function
    def entropy(self):
        return self._entropy

    @property
    def auc(self):
        return self._Z

    @staticmethod
    def _little_phi(x):
        return (-(x ** 2) * 0.5).exp() * CONST_INV_SQRT_2PI

    @staticmethod
    def _big_phi(x):
        return 0.5 * (1 + (x * CONST_INV_SQRT_2).erf())

    @staticmethod
    def _inv_big_phi(x):
        return CONST_SQRT_2 * (2 * x - 1).erfinv()

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return ((self._big_phi(value) - self._big_phi_a) / self._Z).clamp(0, 1)

    def icdf(self, value):
        return self._inv_big_phi(self._big_phi_a + value * self._Z)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return CONST_LOG_INV_SQRT_2PI - self._log_Z - (value ** 2) * 0.5

    def rsample(self, sample_shape=torch.Size()):
        # icdf is numerically unstable; as a consequence, so is rsample.
        shape = self._extended_shape(sample_shape)
        p = torch.empty(shape, device=self.a.device).uniform_(self._dtype_min_gt_0, self._dtype_max_lt_1)
        return self.icdf(p)


class TruncatedNormal(TruncatedStandardNormal):
    """
    Truncated Normal distribution
    https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    """

    has_rsample = True

    def __init__(self, loc, scale, scalar_a, scalar_b, validate_args=True):
        self.loc, self.scale, a, b = broadcast_all(loc, scale, scalar_a, scalar_b)
        a = (a - self.loc) / self.scale
        b = (b - self.loc) / self.scale
        super(TruncatedNormal, self).__init__(a, b, validate_args=validate_args)
        self._log_scale = self.scale.log()
        self._mean = self._mean * self.scale + self.loc
        self._mode = torch.clamp(self.loc, scalar_a, scalar_b)  # pull request?
        self._variance = self._variance * self.scale ** 2
        self._entropy += self._log_scale

    def _to_std_rv(self, value):
        return (value - self.loc) / self.scale

    def _from_std_rv(self, value):
        return value * self.scale + self.loc

    def cdf(self, value):
        return super(TruncatedNormal, self).cdf(self._to_std_rv(value))

    def icdf(self, value):
        return self._from_std_rv(super(TruncatedNormal, self).icdf(value))

    def log_prob(self, value):
        return super(TruncatedNormal, self).log_prob(self._to_std_rv(value)) - self._log_scale


#
class TruncNormalDist(TruncatedNormal):

    has_rsample=True
    def __init__(self, loc, scale, low, high, clip=1e-6, mult=1):
        super().__init__(loc, scale, low, high)
        self._clip = clip
        self._mult = mult

        self.low = low
        self.high = high

    def rsample(self, *args, **kwargs):
        event = super().rsample(*args, **kwargs)
        if self._clip:
            # clipped = tf.clip_by_value(
            #     event, self.low + self._clip, self.high - self._clip)
            # event = event - tf.stop_gradient(event) + tf.stop_gradient(clipped)

            clipped = torch.clamp(event, self.low + self._clip, self.high - self._clip)
            event = event - event.detach() + clipped.detach()
        if self._mult:
            event *= self._mult
        return event


class DistLayer(nn.Module):

  def __init__(
      self, shape, in_dim, dist='mse', min_std=0.1, init_std=0.0):
    super().__init__()

    self._shape = shape
    if isinstance(shape, int):
      self._shape = [shape]

    self._dist = dist
    self._min_std = min_std
    self._init_std = init_std

    if in_dim:
      self._mean_net = nn.Linear(in_dim, np.prod(self._shape))

      if self._dist in ('normal', 'tanh_normal', 'trunc_normal'):
        self._std_net = nn.Linear(in_dim, np.prod(self._shape))
      
  def forward(self, inputs, flatten=False):
    if hasattr(self, '_mean_net'):
      out = self._mean_net(inputs)
      out = torch.reshape(out, list(inputs.shape[:-1]) + self._shape)
    else:
      out = inputs

    if hasattr(self, '_std_net'):
      std = self._std_net(inputs)
      std = torch.reshape(std, list(inputs.shape[:-1]) + self._shape)
    # FIXME: Fix the shapes later
    if self._dist == 'mse':
      dist = td.Normal(out, torch.ones_like(out))
      return dist
    #   return td.independent.Independent(dist, len(self._shape))
    if self._dist == 'normal':
      std = nn.functional.softplus(std + self._init_std) + self._min_std
      dist = td.Normal(out, std)
      return dist
    #   return td.independent.Independent(dist, len(self._shape))
    if self._dist == 'binary':
      dist = td.Bernoulli(logits=out)
      return dist
    #   return td.independent.Independent(dist, len(self._shape))
    if self._dist == 'trunc_normal':
      std = 2 * nn.functional.sigmoid((std + self._init_std) / 2) + self._min_std
      dist = TruncNormalDist(nn.functional.tanh(out), std, -1, 1)
      return dist #td.independent.Independent(dist, 1)
    if self._dist == 'onehot':
      return td.OneHotCategoricalStraightThrough(logits=out)
    raise NotImplementedError(self._dist)
