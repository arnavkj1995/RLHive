from hive.utils.registry import registry
from hive.agents.qnets.atari import NatureAtariDQNModel
from hive.agents.qnets.base import FunctionApproximator
from hive.agents.qnets.conv import ConvNetwork
from hive.agents.qnets.mlp import MLPNetwork
from hive.agents.qnets.deconv import DeconvNetwork
from hive.agents.qnets.wm import WorldModel
from hive.agents.qnets.rssm import RSSM
from hive.agents.qnets.ac import ActorCritic
from hive.agents.qnets.dists import DistLayer

registry.register_all(
    FunctionApproximator,
    {
        "ConvNetwork": ConvNetwork,
        "DeconvNetwork": DeconvNetwork,
        "NatureAtariDQNModel": NatureAtariDQNModel,
        "WorldModel": WorldModel,
        "RSSM": RSSM,
        "ActorCritic": ActorCritic,
        "DistLayer": DistLayer,
    },
)

get_qnet = getattr(registry, f"get_{FunctionApproximator.type_name()}")
