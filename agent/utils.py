import numpy as np
from gym import spaces
from habitat.core.spaces import ActionSpace, EmptySpace

def make_spaces_from_ckpt_config(cfg_dict):
    """
    Build observation_space and action_space from a Habitat/Habitat-Baselines
    config dict (like ckpt['config']), without creating an Env.
    """
    habitat_cfg = cfg_dict["habitat"]

    # ---------- OBSERVATION SPACE ----------
    sim_cfg = habitat_cfg["simulator"]
    task_cfg = habitat_cfg["task"]

    sim_sensors = sim_cfg["agents"]["main_agent"]["sim_sensors"]
    lab_sensors = task_cfg.get("lab_sensors", {})

    obs_spaces = {}

    # 1) Simulator sensors: rgb + depth
    for sensor_name, s_cfg in sim_sensors.items():
        h = s_cfg["height"]
        w = s_cfg["width"]
        sensor_type = s_cfg["type"]

        # RGB sensor
        if "RGBSensor" in sensor_type:
            obs_spaces["rgb"] = spaces.Box(
                low=0,
                high=255,
                shape=(h, w, 3),
                dtype=np.uint8,
            )

        # You can extend here for semantic sensors, etc.

    # 2) Task sensors: ImageGoal, Compass, GPS
    for sensor_name, s_cfg in lab_sensors.items():
        stype = s_cfg["type"]

        # ImageGoalSensor: same shape as rgb
        if stype == "ImageGoalSensor":
            # assume same res as main rgb_sensor
            rgb_cfg = sim_sensors["rgb_sensor"]
            h = rgb_cfg["height"]
            w = rgb_cfg["width"]
            obs_spaces["imagegoal"] = spaces.Box(
                low=0,
                high=255,
                shape=(h, w, 3),
                dtype=np.uint8,
            )

        elif stype == "CompassSensor":
            # Standard Habitat: single angle in [-pi, pi]
            obs_spaces["compass"] = spaces.Box(
                low=-np.pi,
                high=np.pi,
                shape=(1,),
                dtype=np.float32,
            )

        elif stype == "GPSSensor":
            dim = s_cfg.get("dimensionality", 2)
            obs_spaces["gps"] = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(dim,),
                dtype=np.float32,
            )

        # Add more custom lab sensors here if needed.

    observation_space = spaces.Dict(obs_spaces)

    # ---------- ACTION SPACE ----------
    actions_cfg = task_cfg["actions"]
    # For Nav-v0, all actions are discrete; space size is just number of actions.
    # IMPORTANT: ordering of actions matters if you use a real env; here we just
    # care about size, which is enough for actor_critic construction.
    action_names = list(actions_cfg.keys())
    action_space = spaces.Discrete(len(action_names))
    habitat_action_space = ActionSpace({name: EmptySpace() for name in actions_cfg.keys()})

    return observation_space, habitat_action_space
