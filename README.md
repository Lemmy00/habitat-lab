# Training and Comparing Depth-only vs. Depth+RGB Policies in Habitat-Lab

This guide explains how to set up and train navigation policies in Habitat-Lab using either depth information only or both depth and RGB data, to evaluate how much RGB imagery contributes to navigation performance.

## Step 1: Clone the Repository and Set Up Habitat-Lab

Clone the repository and follow the instructions to install Habitat-Lab dependencies as specified in the `HABITAT.md` file:

```bash
git clone --branch depth-only-img-nav --single-branch https://github.com/Lemmy00/habitat-lab.git
cd habitat-lab
```

Please follow the installation steps outlined in `HABITAT.md` to install the necessary dependencies.

> ⚠️ Note: You will need to adjust the path to your Miniconda installation inside the Slurm scripts (`srun_policy_train.sh` and `srun_policy_train_depth_only.sh`). All Slurm scripts must be executed from the **root directory** of the `habitat-lab` repository.

## Step 2: Download and Prepare the Dataset

From the root of the Habitat-Lab directory, run the following commands:

```bash
mkdir -p data/datasets/pointnav/hm3d/v2
cd data/datasets/pointnav/hm3d/v2

wget https://dl.fbaipublicfiles.com/habitat/data/datasets/imagenav/hm3d/v2/instance_imagenav_hm3d_v2.zip

unzip instance_imagenav_hm3d_v2.zip
cd ../../../..
```

## Step 3: Train Navigation Policies

You can now train two types of policies:

### Depth-only Policy

```bash
python rl-distance-train/train_image_nav.py --depth-only --dist-to-goal
```

### RGB + Depth Policy

```bash
python rl-distance-train/train_image_nav.py --dist-to-goal
```

> Note: The `--dist-to-goal` flag enables the agent to use the relative distance to the goal as an additional observation.

## Step 4: Launching with Slurm

To train using Slurm job on Euler, submit the corresponding job scripts from the root of the repo:

```bash
sbatch < srun_policy_train.sh
sbatch < srun_policy_train_depth_only.sh
```

Make sure to edit the job scripts and replace the Miniconda activation path with your own path.

## Step 5: Evaluating and Comparing Policies

After training, evaluate the performance of both policies using the TensorBoard logs and the provided evaluation notebook (refer to the `rl-distance-train` directory).

For advanced parameter tuning, consult the following YAML config files:

- `habitat-baselines/habitat_baselines/config/instance_imagenav/ddppo_instance_imagenav.yaml`
- `habitat/config/benchmark/nav/instance_imagenav/instance_imagenav_hm3d_v2.yaml`
