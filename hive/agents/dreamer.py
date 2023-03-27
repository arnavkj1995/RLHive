import copy
import os

import gym
import numpy as np
import torch
from torch.nn.functional import mse_loss
import torch.distributions as td

from hive.agents.agent import Agent
from hive.agents.qnets.wm import WorldModel
from hive.agents.qnets.ac import ActorCritic
from hive.agents.qnets.base import FunctionApproximator
from hive.agents.qnets.td3_heads import TD3ActorNetwork, TD3CriticNetwork
from hive.agents.qnets.utils import (
    InitializationFn,
    calculate_output_dim,
    create_init_weights_fn,
)
from hive.replays import BaseReplayBuffer, CircularReplayBuffer, replay_buffer
from hive.utils.loggers import Logger, NullLogger
from hive.utils.schedule import PeriodicSchedule, SwitchSchedule
from hive.utils.utils import LossFn, OptimizerFn, create_folder

def action_noise(action, amount, act_space):
    if amount == 0:
        return action
    # amount = tf.cast(amount, action.dtype)
    if hasattr(act_space, 'n'):
        probs = amount / action.shape[-1] + (1 - amount) * action
        return td.OneHotDistWithStraightThrough(probs=probs).sample()
    else:
        return torch.clamp(td.normal.Normal(action, amount).sample(), -1, 1)

class Random(torch.nn.Module):

    def __init__(self, action_dim):
        self._action_dim = action_dim

    def actor(self, feat, discrete=True):
        shape = list(feat.shape[:-1]) + [self._action_dim]
        # return td.OneHotCategorical(logits=torch.zeros(shape))
        if discrete:
            return td.OneHotCategoricalStraightThrough(torch.zeros(shape))
        else:
            dist = td.Uniform(-torch.ones(shape), torch.ones(shape))
            return td.Independent(dist, 1)

    def train(self, start, context, data):
        return None, {}

class Dreamer(Agent):

    """An agent implementing the TD3 algorithm."""
    def __init__(
        self,
        observation_space: gym.spaces.Box,
        action_space: gym.spaces.Box,
        wm: FunctionApproximator = None,
        policy: FunctionApproximator = None,
        # init_fn: InitializationFn = None,
        stack_size: int = 1,
        batch_size: int = 64,
        batch_length: int = 50,
        train_every: int = 5,
        train_steps: int = 1,
        replay_buffer: BaseReplayBuffer = None,
        # discount_rate: float = 0.99,
        # n_step: int = 1,
        # grad_clip: float = None,
        reward_clip: str = None,
        # soft_update_fraction: float = 0.005,
        # batch_size: int = 64,
        logger: Logger = None,
        expl_behavior: str = 'random',
        expl_noise: float = 0.0,
        eval_noise: float = 0.0,
        eval_state_mean: float = 0.0,
        log_frequency: int = 100,
        vid_log_frequency: int = 2000,
        min_replay_history: int = 0,
        # update_frequency: int = 1,
        # policy_update_frequency: int = 2,
        # action_noise: float = 0,
        # target_noise: float = 0.2,
        # target_noise_clip: float = 0.5,
        # min_replay_history: int = 1000,
        device="cuda:0",
        id=0,
    ):
        """
        Args:
            observation_space (gym.spaces.Box): Observation space for the agent.
            action_space (gym.spaces.Box): Action space for the agent.
            representation_net (FunctionApproximator): The network that encodes the
                observations that are then fed into the actor_net and critic_net. If
                None, defaults to :py:class:`~torch.nn.Identity`.
            actor_net (FunctionApproximator): The network that takes the encoded
                observations from representation_net and outputs the representations
                used to compute the actions (ie everything except the last layer).
            critic_net (FunctionApproximator): The network that takes two inputs: the
                encoded observations from representation_net and actions. It outputs
                the representations used to compute the values of the actions (ie
                everything except the last layer).
            init_fn (InitializationFn): Initializes the weights of agent networks using
                create_init_weights_fn.
            actor_optimizer_fn (OptimizerFn): A function that takes in the list of
                parameters of the actor returns the optimizer for the actor. If None,
                defaults to :py:class:`~torch.optim.Adam`.
            critic_optimizer_fn (OptimizerFn): A function that takes in the list of
                parameters of the critic returns the optimizer for the critic. If None,
                defaults to :py:class:`~torch.optim.Adam`.
            critic_loss_fn (LossFn): The loss function used to optimize the critic. If
                None, defaults to :py:class:`~torch.nn.MSELoss`.
            n_critics (int): The number of critics used by the agent to estimate
                Q-values. The minimum Q-value is used as the value for the next state
                when calculating target Q-values for the critic. The output of the
                first critic is used when computing the loss for the actor. For TD3,
                the default value is 2. For DDPG, this parameter is 1.
            stack_size (int): Number of observations stacked to create the state fed
                to the agent.
            replay_buffer (BaseReplayBuffer): The replay buffer that the agent will
                push observations to and sample from during learning. If None,
                defaults to
                :py:class:`~hive.replays.circular_replay.CircularReplayBuffer`.
            discount_rate (float): A number between 0 and 1 specifying how much
                future rewards are discounted by the agent.
            n_step (int): The horizon used in n-step returns to compute TD(n) targets.
            grad_clip (float): Gradients will be clipped to between
                [-grad_clip, grad_clip].
            reward_clip (float): Rewards will be clipped to between
                [-reward_clip, reward_clip].
            soft_update_fraction (float): The weight given to the target
                net parameters in a soft (polyak) update. Also known as tau.
            batch_size (int): The size of the batch sampled from the replay buffer
                during learning.
            logger (Logger): Logger used to log agent's metrics.
            log_frequency (int): How often to log the agent's metrics.
            update_frequency (int): How frequently to update the agent. A value of 1
                means the agent will be updated every time update is called.
            policy_update_frequency (int): Relative update frequency of the actor
                compared to the critic. The actor will be updated every
                policy_update_frequency times the critic is updated.
            action_noise (float): The standard deviation for the noise added to the
                action taken by the agent during training.
            target_noise (float): The standard deviation of the noise added to the
                target policy for smoothing.
            target_noise_clip (float): The sampled target_noise is clipped to
                [-target_noise_clip, target_noise_clip].
            min_replay_history (int): How many observations to fill the replay buffer
                with before starting to learn.
            device: Device on which all computations should be run.
            id: Agent identifier.
        """
        super().__init__(observation_space, action_space, id)
        self._device = torch.device("cpu" if not torch.cuda.is_available() else device)
        
        # self._config = config
        self._obs_space = observation_space
        self._act_space = action_space
        self._reward_clip = reward_clip
        self._batch_size = batch_size
        self._batch_length = batch_length
        self._train_every = train_every
        self._train_steps = train_steps
        
        # self._step = step
        # FIXME: Is there a way to replace tfstep in Hive or do we need to do it?
        # self.tfstep = tf.Variable(int(self.step), tf.int64)
        
        discrete = hasattr(self._act_space, 'n')
        self._action_dim = self._act_space.n if discrete else self._act_space.shape[0]

        self._wm = wm(observation_space, self._action_dim, device).to(self._device) #WorldModel(config, observation_space, self.tfstep)
        # FIXME: Define the ActorCritic network at the top
        if self._wm.rssm._discrete:
            feat_dim = self._wm.rssm._deter + self._wm.rssm._stoch * self._wm.rssm._discrete
        else:
            feat_dim = self._wm.rssm._deter + self._wm.rssm._stoch
        self._task_behavior = policy(self._action_dim, feat_dim, discrete).to(self._device)
        
        self._expl_noise = expl_noise
        self._eval_noise = eval_noise
        self._eval_state_mean = eval_state_mean
        if expl_behavior == 'greedy':
            self._expl_behavior = self._task_behavior
        else:
            #     self.config, self.act_space, self.wm, self.tfstep,
            # self._expl_behavior = getattr(expl, config.expl_behavior)(
            #     lambda seq: self.wm.heads['reward'](seq['feat']).mode())
            self._expl_behavior = Random(self._action_dim)

        self._replay_buffer = replay_buffer(
            observation_shape=self._obs_space.shape,
            observation_dtype=self._obs_space.dtype,
            action_shape=self._action_dim,
            action_dtype=self._action_space.dtype,
            # gamma=discount_rate,
        )

        # self._state = {"state": None}

        self._logger = logger
        if self._logger is None:
            self._logger = NullLogger([])
        self._timescale = self.id
        self._logger.register_timescale(
            self._timescale, PeriodicSchedule(False, True, log_frequency)
        )

        # print (self._action_space.low, self._action_space.high)
        # self.random_actor = td.uniform.Uniform(self._action_space.low, self._action_space.high)

        self._update_period_schedule = PeriodicSchedule(False, True, train_every)
        self._vidlog_schedule = PeriodicSchedule(False, True, vid_log_frequency)
        self._learn_schedule = SwitchSchedule(False, True, min_replay_history)

    def train(self):
        """Changes the agent to training mode."""        
        super().train()
        self._wm.train()
        self._task_behavior.train()

    def eval(self):
        """Changes the agent to evaluation mode."""
        super().eval()
        self._wm.eval()
        self._task_behavior.eval()

    def preprocess_update_info(self, update_info):
        """Preprocesses the :obj:`update_info` before it goes into the replay buffer.
        Scales the action to [-1, 1].

        Args:
            update_info: Contains the information from the current timestep that the
                agent should use to update itself.
        """
        if self._reward_clip is not None:
            update_info['reward'] = {
                'sign': np.sign,
                'tanh': np.tanh
            }[self._reward_clip](update_info['reward'])

        preprocessed_update_info = {
            "observation": update_info["observation"],
            "action": update_info["action"], #self.scale_action(update_info["action"]),
            "reward": update_info["reward"],
            "done": update_info["done"],
        }
        if "agent_id" in update_info:
            preprocessed_update_info["agent_id"] = int(update_info["agent_id"])

        return preprocessed_update_info

    def preprocess_update_batch(self, batch):
        """Preprocess the batch sampled from the replay buffer.

        Args:
            batch: Batch sampled from the replay buffer for the current update.

        Returns:
            (tuple):
                - (tuple) Inputs used to calculate current state values.
                - (tuple) Inputs used to calculate next state values
                - Preprocessed batch.
        """
        for key in batch:
            batch[key] = torch.tensor(batch[key], device=self._device)
        return batch #(batch["observation"],), (batch["next_observation"],), batch

    @torch.no_grad()
    def act(self, observation, state=None):
        """Returns the action for the agent. If in training mode, adds noise with
        standard deviation :py:obj:`self._action_noise`.

        Args:
            observation: The current observation.
        """

        # FIXME: Check why the nest.map_structure is called. 
        # obs = tf.nest.map_structure(tf.tensor, obs)
        # tf.py_function(lambda: self.tfstep.assign(
        #     int(self.step), read_value=False), [], [])
        
        if self._learn_schedule.get_value():
            outputs, state = self.act_imagine(torch.tensor([observation]), state)
            return outputs["action"][0].cpu().numpy(), state

        return np.random.uniform(-1, 1, size=self._action_space.shape), state        

    def act_imagine(self, observation, state=None):
        """Returns the action for the agent. If in training mode, adds noise with
        standard deviation :py:obj:`self._action_noise`.

        Args:
            observation: The current observation.
        """

        # FIXME: Check why the nest.map_structure is called. 
        # obs = tf.nest.map_structure(tf.tensor, obs)
        # tf.py_function(lambda: self.tfstep.assign(
        #     int(self.step), read_value=False), [], [])
        batch = {"observation": observation.to(self._device), "reward": torch.tensor([0.0]).to(self._device), "is_first": torch.tensor([False]).to(self._device)}
        if state is None:
            latent = self._wm.rssm.initial(1) #len(observation['reward']))
            action = torch.zeros([1, self._action_dim]).to(self._device) #(len(observation['reward']),) + self.act_space.shape)
            state = latent, action
        latent, action = state
        batch = self._wm.preprocess(batch)
        embed = self._wm.encoder(batch["observation"])
        
        # FIXME: Add the reshape part later
        shape = list(embed.shape[:-3]) + [-1]
        embed = embed.view(shape)      
        
        # FIXME: Fix the mode issue pytorch
        sample = self._training or not self._eval_state_mean
        latent, _ = self._wm.rssm.obs_step(
            latent, action, embed, batch['is_first'], sample)
        feat = self._wm.rssm.get_feat(latent)

        if not self._training:
            actor = self._task_behavior.actor(feat)
            action = actor.mean
            noise = self._eval_noise
        else:
            # print (" Before actor ", feat.shape)
            actor = self._task_behavior.actor(feat)
            action = actor.rsample()
            noise = self._expl_noise
            
        # FIXME: Add action_noise to the utils
        action = action_noise(action, noise, self._act_space)
        outputs = {'action': action}
        state = (latent, action)

        return outputs, state

    def save_gif(self, images):
        import imageio
        print (images.shape)
        imageio.mimsave('movie.gif', np.transpose(images, (0, 2, 3, 1)), fps=10)
        sys.exit(-1)

    def update(self, update_info, state=None):
        """
        Updates the TD3 agent.

        Args:
            update_info: dictionary containing all the necessary information to
                update the agent. Should contain a full transition, with keys for
                "observation", "action", "reward", and "done".
        """
        
                
        # Add the most recent transition to the replay buffer.
        self._replay_buffer.add(**self.preprocess_update_info(update_info))

        if self._learn_schedule.update() and self._update_period_schedule.update():
            batch = self.preprocess_update_batch(self._replay_buffer.sample(self._batch_size, self._batch_length))

            batch["reward"] = batch["reward"].unsqueeze(-1)
            batch["done"] = batch["done"].unsqueeze(-1)

            metrics = {}
            # FIXME: Check what the state should be here
            state, outputs, mets = self._wm.update(batch, None) #state)
            metrics.update(mets)

            start = outputs['post']
            reward = lambda seq: self._wm.heads['reward'](seq['feat']).mean #, flatten=False) #.get_mode()

            # FIXME: Set the is_terminal flag here
            is_terminal = False
            metrics.update(self._task_behavior.update(
                self._wm, start, is_terminal, reward))

            if self._logger.update_step(self._timescale):        
                self._logger.log_metrics(metrics, self._timescale)

            if self._vidlog_schedule.update():
                data = self._wm.preprocess(batch)
                vid = self._wm.video_pred(data, 'observation') #, (0, 2, 3, 1))
                vid = np.clip(255 * vid, 0, 255).astype(np.uint8)
                self._logger.log_gif('WM', vid , self._timescale)

            # if self.config.expl_behavior != 'greedy':
            #     mets = self._expl_behavior.train(start, outputs, data)[-1]
            #     metrics.update({'expl_' + key: value for key, value in mets.items()})

    def save(self, dname):
        pass
        torch.save(
            {
                "critic": self._critic.state_dict(),
                "target_critic": self._target_critic.state_dict(),
                "critic_optimizer": self._critic_optimizer.state_dict(),
                "actor": self._actor.state_dict(),
                "target_actor": self._target_actor.state_dict(),
                "actor_optimizer": self._actor_optimizer.state_dict(),
                "learn_schedule": self._learn_schedule,
                "update_schedule": self._update_schedule,
                "policy_update_schedule": self._policy_update_schedule,
            },
            os.path.join(dname, "agent.pt"),
        )
        replay_dir = os.path.join(dname, "replay")
        create_folder(replay_dir)
        self._replay_buffer.save(replay_dir)

    def load(self, dname):
        pass
        checkpoint = torch.load(os.path.join(dname, "agent.pt"))
        self._critic.load_state_dict(checkpoint["critic"])
        self._target_critic.load_state_dict(checkpoint["target_critic"])
        self._critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self._actor.load_state_dict(checkpoint["actor"])
        self._target_actor.load_state_dict(checkpoint["target_actor"])
        self._actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self._learn_schedule = checkpoint["learn_schedule"]
        self._update_schedule = checkpoint["update_schedule"]
        self._policy_update_schedule = checkpoint["policy_update_schedule"]
        self._replay_buffer.load(os.path.join(dname, "replay"))
