#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import abc
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
)

import torch
from gym import spaces
from torch import nn as nn

from rl_distance_train.common import (
    CategoricalNet,
    GaussianNet,
    get_num_actions,
)

if TYPE_CHECKING:
    from omegaconf import DictConfig

from torch import Tensor


@dataclass
class PolicyActionData:
    """
    Information returned from the `Policy.act` method representing the
    information from an agent's action.

    :property should_inserts: Of shape [# envs, 1]. If False at environment
        index `i`, then don't write this transition to the rollout buffer. If
        `None`, then write all data.
    :property policy_info`: Optional logging information about the policy per
        environment. For example, you could log the policy entropy.
    :property take_actions`: If specified, these actions will be executed in
        the environment, but not stored in the storage buffer. This allows
        exectuing and learning from different actions. If not specified, the
        agent will execute `self.actions`.
    :property values: The actor value predictions. None if the actor does not predict value.
    :property actions: The actions to store in the storage buffer. if
        `take_actions` is None, then this is also the action executed in the
        environment.
    :property rnn_hidden_states: Actor hidden states.
    :property action_log_probs: The log probabilities of the actions under the
        current policy.
    """

    rnn_hidden_states: Optional[torch.Tensor] = None
    actions: Optional[torch.Tensor] = None
    values: Optional[torch.Tensor] = None
    action_log_probs: Optional[torch.Tensor] = None
    take_actions: Optional[torch.Tensor] = None
    policy_info: Optional[List[Dict[str, Any]]] = None
    should_inserts: Optional[torch.BoolTensor] = None

    def write_action(self, write_idx: int, write_action: torch.Tensor) -> None:
        """
        Used to override an action across all environments.
        :param write_idx: The index in the action dimension to write the new action.
        :param write_action: The action to write at `write_idx`.
        """
        self.actions[:, write_idx] = write_action

    @property
    def env_actions(self) -> torch.Tensor:
        """
        The actions to execute in the environment.
        """

        if self.take_actions is None:
            return self.actions
        else:
            return self.take_actions


class Policy(abc.ABC):
    def __init__(self, action_space):
        self._action_space = action_space

    @property
    def should_load_agent_state(self):
        return True

    @property
    def hidden_state_shape(self):
        """
        Stack the hidden states of all the policies in the active population.
        """
        raise NotImplementedError(
            "hidden_state_shape is only supported in neural network policies"
        )

    @property
    def hidden_state_shape_lens(self):
        """
        Stack the hidden states of all the policies in the active population.
        """
        raise NotImplementedError(
            "hidden_state_shape_lens is only supported in neural network policies"
        )

    @property
    def policy_action_space_shape_lens(self) -> List[int]:
        return [self._action_space]

    @property
    def policy_action_space(self) -> spaces.Space:
        return self._action_space

    @property
    def num_recurrent_layers(self) -> int:
        return 0

    @property
    def recurrent_hidden_size(self) -> int:
        return 0

    @property
    def visual_encoder(self) -> Optional[nn.Module]:
        """
        Gets the visual encoder for the policy. Only necessary to implement if
        you want to do RL with a frozen visual encoder.
        """

    def update_hidden_state(
        self,
        rnn_hxs: torch.Tensor,
        prev_actions: torch.Tensor,
        action_data: PolicyActionData,
    ) -> None:
        """
        Update the hidden state given that `should_inserts` is not None. Writes
        to `rnn_hxs` and `prev_actions` in place.
        """

        for env_i, should_insert in enumerate(action_data.should_inserts):
            if should_insert.item():
                rnn_hxs[env_i] = action_data.rnn_hidden_states[env_i]
                prev_actions[env_i].copy_(action_data.actions[env_i])  # type: ignore

    def _get_policy_components(self) -> List[nn.Module]:
        return []

    def aux_loss_parameters(self) -> Dict[str, Iterable[torch.Tensor]]:
        """
        Gets parameters of auxiliary modules, not directly used in the policy,
        but used for auxiliary training objectives. Only necessary if using
        auxiliary losses.
        """
        return {}

    def policy_parameters(self) -> Iterable[torch.Tensor]:
        for c in self._get_policy_components():
            yield from c.parameters()

    def all_policy_tensors(self) -> Iterable[torch.Tensor]:
        yield from self.policy_parameters()
        for c in self._get_policy_components():
            yield from c.buffers()

    def get_value(
        self, observations, rnn_hidden_states, prev_actions, masks
    ) -> torch.Tensor:
        raise NotImplementedError(
            "Get value is supported in non-neural network policies."
        )

    def get_extra(
        self, action_data: PolicyActionData, infos, dones
    ) -> List[Dict[str, float]]:
        """
        Gets the log information from the policy at the current time step.
        Currently only called during evaluation. The return list should be
        empty for no logging or a list of size equal to the number of
        environments.
        """
        if action_data.policy_info is None:
            return []
        else:
            return action_data.policy_info

    def evaluate_actions(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        action,
        rnn_build_seq_info: Dict[str, torch.Tensor],
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Dict[str, Tensor]]:
        """
        Only necessary to implement if performing RL training with the policy.

        :returns: Tuple containing
            - Predicted value.
            - Log probabilities of actions.
            - Action distribution entropy.
            - RNN hidden states.
            - Auxiliary module losses.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
    ) -> PolicyActionData:
        raise NotImplementedError

    def on_envs_pause(self, envs_to_pause: List[int]) -> None:
        """
        Clean up data when envs are finished. Makes sure that relevant variables
        of a policy are updated as environments pause. This is needed when
        evaluating policies with multiple environments, where some environments
        will run out of episodes to evaluate and will be closing.
        """

    @classmethod
    @abc.abstractmethod
    def from_config(cls, config, observation_space, action_space, **kwargs):
        pass


class NetPolicy(nn.Module, Policy):
    aux_loss_modules: nn.ModuleDict
    action_distribution: nn.Module

    def __init__(
        self, net, action_space, policy_config=None, aux_loss_config=None
    ):
        Policy.__init__(self, action_space)
        nn.Module.__init__(self)
        self.net = net
        self.dim_actions = get_num_actions(action_space)
        self.action_distribution: Union[CategoricalNet, GaussianNet]

        if policy_config is None:
            self.action_distribution_type = "categorical"
        else:
            self.action_distribution_type = (
                policy_config.action_distribution_type
            )

        if self.action_distribution_type == "categorical":
            self.action_distribution = CategoricalNet(
                self.net.output_size, self.dim_actions
            )
        elif self.action_distribution_type == "gaussian":
            self.action_distribution = GaussianNet(
                self.net.output_size,
                self.dim_actions,
                policy_config.action_dist,
            )
        else:
            raise ValueError(
                f"Action distribution {self.action_distribution_type}"
                "not supported."
            )

        self.critic = CriticHead(self.net.output_size)

        self.aux_loss_modules = get_aux_modules(
            aux_loss_config, action_space, self.net
        )

    @property
    def hidden_state_shape(self):
        return (
            self.num_recurrent_layers,
            self.recurrent_hidden_size,
        )

    @property
    def hidden_state_shape_lens(self):
        return [self.recurrent_hidden_size]

    @property
    def recurrent_hidden_size(self) -> int:
        return self.net.recurrent_hidden_size

    @property
    def visual_encoder(self) -> Optional[nn.Module]:
        return self.net.visual_encoder

    @property
    def should_load_agent_state(self):
        return True

    @property
    def num_recurrent_layers(self) -> int:
        return self.net.num_recurrent_layers

    def forward(self, *x):
        raise NotImplementedError

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
    ):
        features, rnn_hidden_states, _ = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        distribution = self.action_distribution(features)
        value = self.critic(features)

        if deterministic:
            if self.action_distribution_type == "categorical":
                action = distribution.mode()
            elif self.action_distribution_type == "gaussian":
                action = distribution.mean
        else:
            action = distribution.sample()

        action_log_probs = distribution.log_probs(action)
        return PolicyActionData(
            values=value,
            actions=action,
            action_log_probs=action_log_probs,
            rnn_hidden_states=rnn_hidden_states,
        )

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks):
        features, _, _ = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        return self.critic(features)

    def evaluate_actions(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        action,
        rnn_build_seq_info: Dict[str, torch.Tensor],
    ):
        features, rnn_hidden_states, aux_loss_state = self.net(
            observations,
            rnn_hidden_states,
            prev_actions,
            masks,
            rnn_build_seq_info,
        )
        distribution = self.action_distribution(features)
        value = self.critic(features)

        action_log_probs = distribution.log_probs(action)
        distribution_entropy = distribution.entropy()

        batch = dict(
            observations=observations,
            rnn_hidden_states=rnn_hidden_states,
            prev_actions=prev_actions,
            masks=masks,
            action=action,
            rnn_build_seq_info=rnn_build_seq_info,
        )
        aux_loss_res = {
            k: v(aux_loss_state, batch)
            for k, v in self.aux_loss_modules.items()
        }

        return (
            value,
            action_log_probs,
            distribution_entropy,
            rnn_hidden_states,
            aux_loss_res,
        )

    def _get_policy_components(self) -> List[nn.Module]:
        return [self.net, self.critic, self.action_distribution]

    def aux_loss_parameters(self) -> Dict[str, Iterable[torch.Tensor]]:
        return {k: v.parameters() for k, v in self.aux_loss_modules.items()}

    @classmethod
    @abc.abstractmethod
    def from_config(cls, config, observation_space, action_space, **kwargs):
        pass


class CriticHead(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)
        nn.init.orthogonal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        return self.fc(x)


class Net(nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        pass

    @property
    @abc.abstractmethod
    def output_size(self):
        pass

    @property
    @abc.abstractmethod
    def num_recurrent_layers(self):
        pass

    @property
    @abc.abstractmethod
    def recurrent_hidden_size(self):
        pass

    @property
    @abc.abstractmethod
    def is_blind(self):
        pass

    @property
    @abc.abstractmethod
    def perception_embedding_size(self) -> int:
        pass



def get_aux_modules(
    aux_loss_config: "DictConfig",
    action_space: spaces.Space,
    net,
) -> nn.ModuleDict:
    aux_loss_modules = nn.ModuleDict()
    if aux_loss_config is None:
        return aux_loss_modules
    raise NotImplementedError("No auxiliary losses implemented yet.")
