from hive.envs.base import BaseEnv, ParallelEnv
from hive.envs.env_spec import EnvSpec
from hive.envs.gym_env import GymEnv

try:
    from hive.envs.minigrid import MiniGridEnv
except ImportError:
    MiniGridEnv = None

try:
    from hive.envs.atari import AtariEnv
except ImportError:
    AtariEnv = None

try:
    from hive.envs.marlgrid import MarlGridEnv
except ImportError:
    MarlGridEnv = None

try:
    from hive.envs.pettingzoo import PettingZooEnv
except ImportError:
    PettingZooEnv = None

try:
    from hive.envs.dmc import DMCEnv
except ImportError:
    DMCEnv = None

from hive.utils.registry import registry

registry.register_all(
    BaseEnv,
    {
        "GymEnv": GymEnv,
        "MiniGridEnv": MiniGridEnv,
        "MarlGridEnv": MarlGridEnv,
        "AtariEnv": AtariEnv,
        "PettingZooEnv": PettingZooEnv,
        "DMCEnv": DMCEnv,
    },
)

get_env = getattr(registry, f"get_{BaseEnv.type_name()}")
