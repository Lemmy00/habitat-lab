#!/usr/bin/env python3
import sys
from pathlib import Path

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
import os
import numpy as np
import torch

from policy import PPOAgent
from rl_distance_train import distance_policy, dataset, distance_policy_gt, geometric_distance_policy

os.environ["TOKENIZERS_PARALLELISM"] = "false"

@torch.no_grad()
def env_navigation(args):
    agent = PPOAgent.from_config(args.ckpt)
    agent.actor_critic.eval()
    agent.reset()

    print("Observation space:", agent.get_observation_space().keys())

    observations = {
        "rgb": np.zeros((256, 256, 3), dtype=np.uint8),
        "imagegoal": np.zeros((256, 256, 3), dtype=np.uint8),
        "compass": np.zeros((1,), dtype=np.float32), # angle in radians of rotation w.r.t the initial heading
        "gps": np.zeros((2,), dtype=np.float32), # (x, y) coordinates w.r.t the start position
    }
    agent_obs = {k: observations[k] for k in agent.get_observation_space().keys()}

    action = agent.act(agent_obs)
    print("Action from initial observation:", action)
    print("Action intepretation:", agent.action_interpretation(action))


def main(args):
    env_navigation(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to trained distance policy checkpoint.")
    args = parser.parse_args()

    main(args)
