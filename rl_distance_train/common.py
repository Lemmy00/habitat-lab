#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import numbers
import shutil
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    OrderedDict,
)

import attr
import numpy as np
import torch
import gym

from gym import spaces
from torch import Size, Tensor
from torch import nn as nn
from gym import Space

from rl_distance_train.tensor_dict import TensorDict, DictTree, TensorOrNDArrayDict

if TYPE_CHECKING:
    from omegaconf import DictConfig


class EmptySpace(Space):
    """
    A ``gym.Space`` that reflects arguments space for action that doesn't have
    arguments. Needed for consistency ang always samples `None` value.
    """

    def sample(self):
        return None

    def contains(self, x):
        if x is None:
            return True
        return False

    def __repr__(self):
        return "EmptySpace()"


class ActionSpace(gym.spaces.Dict):
    """
    A dictionary of ``EmbodiedTask`` actions and their argument spaces.

    .. code:: py

        self.observation_space = spaces.ActionSpace({
            "move": spaces.Dict({
                "position": spaces.Discrete(2),
                "velocity": spaces.Discrete(3)
            }),
            "move_forward": EmptySpace(),
        })
    """

    def __init__(self, spaces: Union[List, Dict]):
        if isinstance(spaces, dict):
            self.spaces = OrderedDict(sorted(spaces.items()))
        if isinstance(spaces, list):
            self.spaces = OrderedDict(spaces)
        self.actions_select = gym.spaces.Discrete(len(self.spaces))

    @property
    def n(self) -> int:
        return len(self.spaces)

    def sample(self):
        action_index = self.actions_select.sample()
        return {
            "action": list(self.spaces.keys())[action_index],
            "action_args": list(self.spaces.values())[action_index].sample(),
        }

    def contains(self, x):
        if not isinstance(x, dict) or "action" not in x:
            return False
        if x["action"] not in self.spaces:
            return False
        if not self.spaces[x["action"]].contains(x.get("action_args", None)):
            return False
        return True

    def __repr__(self):
        return (
            "ActionSpace("
            + ", ".join([k + ":" + str(s) for k, s in self.spaces.items()])
            + ")"
        )

if hasattr(torch, "inference_mode"):
    inference_mode = torch.inference_mode
else:
    inference_mode = torch.no_grad


def cosine_decay(progress: float) -> float:
    progress = min(max(progress, 0.0), 1.0)

    return (1.0 + math.cos(progress * math.pi)) / 2.0


class CustomFixedCategorical(torch.distributions.Categorical):  # type: ignore
    def sample(
        self, sample_shape: Size = torch.Size()  # noqa: B008
    ) -> Tensor:
        return super().sample(sample_shape).unsqueeze(-1)

    def log_probs(self, actions: Tensor) -> Tensor:
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1, keepdim=True)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)

    def entropy(self):
        return super().entropy().unsqueeze(-1)


class CategoricalNet(nn.Module):
    def __init__(self, num_inputs: int, num_outputs: int) -> None:
        super().__init__()

        self.linear = nn.Linear(num_inputs, num_outputs)

        nn.init.orthogonal_(self.linear.weight, gain=0.01)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x: Tensor) -> CustomFixedCategorical:
        x = self.linear(x)
        return CustomFixedCategorical(logits=x.float(), validate_args=False)


class CustomNormal(torch.distributions.normal.Normal):
    def sample(
        self, sample_shape: Size = torch.Size()  # noqa: B008
    ) -> Tensor:
        return self.rsample(sample_shape)

    def log_probs(self, actions) -> Tensor:
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entropy(self) -> Tensor:
        return super().entropy().sum(-1, keepdim=True)


class GaussianNet(nn.Module):
    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        config: "DictConfig",
    ) -> None:
        super().__init__()

        self.action_activation = config.action_activation
        self.use_softplus = config.use_softplus
        self.use_log_std = config.use_log_std
        use_std_param = config.use_std_param
        self.clamp_std = config.clamp_std

        if self.use_log_std:
            self.min_std = config.min_log_std
            self.max_std = config.max_log_std
            std_init = config.log_std_init
        elif self.use_softplus:
            inv_softplus = lambda x: math.log(math.exp(x) - 1)
            self.min_std = inv_softplus(config.min_std)
            self.max_std = inv_softplus(config.max_std)
            std_init = inv_softplus(1.0)
        else:
            self.min_std = config.min_std
            self.max_std = config.max_std
            std_init = 1.0  # initialize std value so that std ~ 1

        if use_std_param:
            self.std = torch.nn.parameter.Parameter(
                torch.randn(num_outputs) * 0.01 + std_init
            )
            num_linear_outputs = num_outputs
        else:
            self.std = None
            num_linear_outputs = 2 * num_outputs

        self.mu_maybe_std = nn.Linear(num_inputs, num_linear_outputs)
        nn.init.orthogonal_(self.mu_maybe_std.weight, gain=0.01)
        nn.init.constant_(self.mu_maybe_std.bias, 0)

        if not use_std_param:
            nn.init.constant_(self.mu_maybe_std.bias[num_outputs:], std_init)

    def forward(self, x: Tensor) -> CustomNormal:
        mu_maybe_std = self.mu_maybe_std(x).float()
        if self.std is not None:
            mu = mu_maybe_std
            std = self.std
        else:
            mu, std = torch.chunk(mu_maybe_std, 2, -1)

        if self.action_activation == "tanh":
            mu = torch.tanh(mu)

        if self.clamp_std:
            std = torch.clamp(std, self.min_std, self.max_std)
        if self.use_log_std:
            std = torch.exp(std)
        if self.use_softplus:
            std = torch.nn.functional.softplus(std)

        return CustomNormal(mu, std, validate_args=False)



def delete_folder(path: str) -> None:
    shutil.rmtree(path)


def action_to_velocity_control(
    action: torch.Tensor,
    allow_sliding: bool = None,
) -> Dict[str, Any]:
    lin_vel, ang_vel = torch.clip(action, min=-1, max=1)
    step_action = {
        "action": {
            "action": "velocity_control",
            "action_args": {
                "linear_velocity": lin_vel.item(),
                "angular_velocity": ang_vel.item(),
                "allow_sliding": allow_sliding,
            },
        }
    }
    return step_action


def iterate_action_space_recursively(action_space):
    if isinstance(action_space, spaces.Dict):
        for v in action_space.values():
            yield from iterate_action_space_recursively(v)
    else:
        yield action_space


def is_continuous_action_space(action_space) -> bool:
    possible_discrete_spaces = (
        spaces.Discrete,
        spaces.MultiDiscrete,
        spaces.Dict,
    )
    if isinstance(action_space, spaces.Box):
        return True
    elif isinstance(action_space, possible_discrete_spaces):
        return False
    else:
        raise NotImplementedError(
            f"Unknown action space {action_space}. Is neither continuous nor discrete"
        )


def get_action_space_info(ac_space: spaces.Space) -> Tuple[Tuple[int], bool]:
    """
    :returns: The shape of the action space and if the action space is discrete. If the action space is discrete, the shape will be `(1,)`.
    """
    if is_continuous_action_space(ac_space):
        # Assume NONE of the actions are discrete
        return (
            (
                get_num_actions(
                    ac_space,
                ),
            ),
            False,
        )

    elif isinstance(ac_space, spaces.MultiDiscrete):
        return ac_space.shape, True
    elif isinstance(ac_space, spaces.Dict):
        num_actions = 0
        for _, ac_sub_space in ac_space.items():
            num_actions += get_action_space_info(ac_sub_space)[0][0]
        return (num_actions,), True

    else:
        # For discrete pointnav
        return (1,), True


def get_num_actions(action_space) -> int:
    num_actions = 0
    for v in iterate_action_space_recursively(action_space):
        if isinstance(v, spaces.Box):
            assert (
                len(v.shape) == 1
            ), f"shape was {v.shape} but was expecting a 1D action"
            num_actions += v.shape[0]
        elif isinstance(v, EmptySpace):
            num_actions += 1
        elif isinstance(v, spaces.Discrete):
            num_actions += v.n
        else:
            raise NotImplementedError(
                f"Trying to count the number of actions with an unknown action space {v}"
            )

    return num_actions


class Singleton(type):
    """
    This metatclass creates Singleton objects by ensuring only one instance is created and any call is directed to that instance. The mro() function and following dunders, EXCEPT __call__, are inherited from the the stdlib Python library, which defines the "type" class.
    """

    _instances: Dict["Singleton", "Singleton"] = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(
                *args, **kwargs
            )
        return cls._instances[cls]


@attr.s(auto_attribs=True, slots=True)
class _ObservationBatchingCache(metaclass=Singleton):
    r"""Helper for batching observations that maintains a cpu-side tensor
    that is the right size and is pinned to cuda memory
    """
    _pool: Dict[Any, Union[torch.Tensor, np.ndarray]] = {}

    def get(
        self,
        num_obs: int,
        sensor_name: Any,
        sensor: torch.Tensor,
        device: Optional[torch.device] = None,
    ) -> Union[torch.Tensor, np.ndarray]:
        r"""Returns a tensor of the right size to batch num_obs observations together

        If sensor is a cpu-side tensor and device is a cuda device the batched tensor will
        be pinned to cuda memory.  If sensor is a cuda tensor, the batched tensor will also be
        a cuda tensor
        """
        key = (
            sensor_name,
            tuple(sensor.size()),
            sensor.type(),
            sensor.device.type,
            sensor.device.index,
        )
        if key in self._pool:
            cache = self._pool[key]
            if cache.shape[0] >= num_obs:
                return cache[0:num_obs]
            else:
                cache = None
                del self._pool[key]

        cache = torch.empty(
            num_obs, *sensor.size(), dtype=sensor.dtype, device=sensor.device
        )
        if (
            device is not None
            and device.type == "cuda"
            and cache.device.type == "cpu"
        ):
            cache = cache.pin_memory()

        if cache.device.type == "cpu":
            # Pytorch indexing is slow,
            # so convert to numpy
            cache = cache.numpy()

        self._pool[key] = cache
        return cache

    def batch_obs(
        self,
        observations: List[DictTree],
        device: Optional[torch.device] = None,
    ) -> TensorDict:
        observations = [
            TensorOrNDArrayDict.from_tree(o).map(
                lambda t: t.numpy()
                if isinstance(t, torch.Tensor) and t.device.type == "cpu"
                else t
            )
            for o in observations
        ]
        observation_keys, _ = observations[0].flatten()
        observation_tensors = [o.flatten()[1] for o in observations]

        # Order sensors by size, stack and move the largest first
        upload_ordering = sorted(
            range(len(observation_keys)),
            key=lambda idx: 1
            if isinstance(observation_tensors[0][idx], numbers.Number)
            else int(np.prod(observation_tensors[0][idx].shape)),  # type: ignore
            reverse=True,
        )

        batched_tensors = []
        for sensor_name, obs in zip(observation_keys, observation_tensors[0]):
            batched_tensors.append(
                self.get(
                    len(observations),
                    sensor_name,
                    torch.as_tensor(obs),
                    device,
                )
            )

        for idx in upload_ordering:
            for i, all_obs in enumerate(observation_tensors):
                obs = all_obs[idx]
                # Use isinstance(sensor, np.ndarray) here instead of
                # np.asarray as this is quickier for the more common
                # path of sensor being an np.ndarray
                # np.asarray is ~3x slower than checking
                if isinstance(obs, np.ndarray):
                    batched_tensors[idx][i] = obs  # type: ignore
                elif isinstance(obs, torch.Tensor):
                    batched_tensors[idx][i].copy_(obs, non_blocking=True)  # type: ignore
                # If the sensor wasn't a tensor, then it's some CPU side data
                # so use a numpy array
                else:
                    batched_tensors[idx][i] = np.asarray(obs)  # type: ignore

            # With the batching cache, we use pinned mem
            # so we can start the move to the GPU async
            # and continue stacking other things with it
            # If we were using a numpy array to do indexing and copying,
            # convert back to torch tensor
            # We know that batch_t[sensor_name] is either an np.ndarray
            # or a torch.Tensor, so this is faster than torch.as_tensor
            if isinstance(batched_tensors[idx], np.ndarray):
                batched_tensors[idx] = torch.from_numpy(batched_tensors[idx])

            batched_tensors[idx] = batched_tensors[idx].to(  # type: ignore
                device, non_blocking=True
            )

        return TensorDict.from_flattened(observation_keys, batched_tensors)

#@profiling_wrapper.RangeContext("batch_obs")


@inference_mode()
def batch_obs(
    observations: List[DictTree],
    device: Optional[torch.device] = None,
) -> TensorDict:
    r"""Transpose a batch of observation dicts to a dict of batched
    observations.

    Args:
        observations:  list of dicts of observations.
        device: The torch.device to put the resulting tensors on.
            Will not move the tensors if None
    Returns:
        transposed dict of torch.Tensor of observations.
    """

    return _ObservationBatchingCache().batch_obs(observations, device)