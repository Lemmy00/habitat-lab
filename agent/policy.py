# Standard library
import random
from typing import Any, Dict, Optional, Union
from enum import Enum

# Third-party
import torch
from gym.spaces import Dict as SpaceDict
from omegaconf import DictConfig, OmegaConf

# Habitat & Habitat Baselines
from agent.utils import make_spaces_from_ckpt_config

from rl_distance_train.common import batch_obs
from rl_distance_train.distance_policy_no_habitat import TemporalDistanceNavPolicy


class HabitatSimActions(Enum):
    """Enum class for default cylinder agent move and look actions. I.e. pointnav action space."""

    stop = 0
    move_forward = 1
    turn_left = 2
    turn_right = 3
    look_up = 4
    look_down = 5


class Agent:
    r"""Abstract class for defining agents which act inside :ref:`core.env.Env`.

    This abstract class standardizes agents to allow seamless benchmarking.
    """

    def reset(self) -> None:
        r"""Called before starting a new episode in environment."""
        raise NotImplementedError

    def act(
        self, observations
    ) -> Union[int, str, Dict[str, Any]]:
        r"""Called to produce an action to perform in an environment.

        :param observations: observations coming in from environment to be
            used by agent to decide action.
        :return: action to be taken inside the environment and optional action
            arguments.
        """
        raise NotImplementedError


class PPOAgent(Agent):
    def __init__(self, config: DictConfig, model_weights: Dict) -> None:
        super().__init__()
        self.config = config
        self.obs_space, self.act_space = make_spaces_from_ckpt_config(OmegaConf.to_container(self.config))

        self.device = (
            torch.device(f"cuda:{self.config.habitat.simulator.habitat_sim_v0.gpu_device_id}")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.hidden_size = self.config.habitat_baselines.rl.ppo.hidden_size

        random.seed(42)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True  # type: ignore

        self.actor_critic = TemporalDistanceNavPolicy.from_config(
            self.config, 
            observation_space=self.obs_space, 
            action_space=self.act_space
        )
        self.actor_critic.to(self.device)
        self.actor_critic.eval()

        self.model_path = model_weights
        if self.model_path:
            self.actor_critic.load_state_dict(model_weights, strict=True)
            print("Loaded model weights successfully.")
        else:
            raise ValueError("Model checkpoint wasn't loaded, evaluating a random model.")

        self.test_recurrent_hidden_states: Optional[torch.Tensor] = None
        self.not_done_masks: Optional[torch.Tensor] = None
        self.prev_actions: Optional[torch.Tensor] = None

    def reset(self) -> None:
        self.test_recurrent_hidden_states = torch.zeros(
            1, self.actor_critic.net.num_recurrent_layers, self.hidden_size, device=self.device,
        )
        self.not_done_masks = torch.zeros(1, 1, device=self.device, dtype=torch.bool)
        self.prev_actions = torch.zeros(1, 1, dtype=torch.long, device=self.device)

    def act(self, observations) -> Dict[str, int]:
        batch = batch_obs([observations], device=self.device)
        batch = {k: v.float() for k, v in batch.items()}
        with torch.no_grad():
            action_data = self.actor_critic.act(
                batch,
                self.test_recurrent_hidden_states,
                self.prev_actions,
                self.not_done_masks,
                deterministic=True,
            )
            self.test_recurrent_hidden_states = action_data.rnn_hidden_states
            self.not_done_masks.fill_(True)
            self.prev_actions.copy_(action_data.actions)

        return action_data.env_actions[0][0].item()
    
    def action_interpretation(self, action: int) -> Dict[str, int]:
        # """Convert action index to a dictionary mapping action names to values."""
        if HabitatSimActions.move_forward.value == action:
            return {"move_forward in meters": self.config.habitat.simulator.forward_step_size}
        elif HabitatSimActions.turn_left.value == action:
            return {"turn_left in degrees": self.config.habitat.simulator.turn_angle}
        elif HabitatSimActions.turn_right.value == action:
            return {"turn_right in degrees": self.config.habitat.simulator.turn_angle}
        elif HabitatSimActions.stop.value == action:
            return {"stop": 1}
        else:
            return {"unknown_action": action}
    
    def get_observation_space(self) -> SpaceDict:
        """Return the observation space of the agent."""
        return self.obs_space

    def get_config(self) -> "DictConfig":
        return self.config

    @staticmethod
    def from_config(config_path: str) -> "PPOAgent":
        """Load the agent configuration from a file or DictConfig."""
        if config_path.endswith((".pth", ".pt")):
            config = torch.load(config_path, map_location="cpu", weights_only=False)
            model_weights = config['state_dict']
            config = OmegaConf.create(config['config'])
        else:
            raise ValueError("Unsupported file format. Use .yaml or .pth/.pt files.")

        return PPOAgent(config, model_weights)
