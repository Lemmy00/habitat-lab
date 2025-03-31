# Training and Comparing Depth-only vs. Depth+RGB Policies in Habitat-Lab

This guide explains how to set up and train navigation policies in Habitat-Lab using either depth information only or both depth and RGB data, to evaluate how much RGB imagery contributes to navigation performance.

## Step 1: Clone the Repository and Set Up Habitat-Lab

Clone the repository and follow the instructions to install Habitat-Lab dependencies as specified in the `HABITAT.md` file:

```bash
git clone --branch depth-only-img-nav --single-branch https://github.com/Lemmy00/habitat-lab.git
cd habitat-lab

# Please then follow the installation steps outlined in HABITAT.md to install the necessary dependencies.
```

## Step 2: Download and Prepare the Dataset

From the root of the Habitat-Lab directory, run the following commands to download and prepare the dataset:

```bash
mkdir -p data/datasets/pointnav/hm3d/v2
cd data/datasets/pointnav/hm3d/v2

wget https://dl.fbaipublicfiles.com/habitat/data/datasets/imagenav/hm3d/v2/instance_imagenav_hm3d_v2.zip

unzip instance_imagenav_hm3d_v2.zip
```

Once done, return to the root directory of the Habitat-Lab repository:

```bash
cd ../../../..
```

## Step 3: Train Navigation Policies

You can now train two types of policies:

### Depth-only Policy
To train a policy using only depth information, execute:

```bash
python rl-distance-train/train_image_nav.py --depth-only
```

### RGB + Depth Policy
To train a policy using both RGB and depth data, execute:

```bash
python rl-distance-train/train_image_nav.py
```

## Step 4: Evaluating and Comparing Policies

After training, you can evaluate the performance of both policies using the produced tensorboard logs and provided evaluation notebook (refer to `rl-distance-train`).

In case you are interested in further parameter tuning please refer to following `.yaml` files:

- `path/to/habitat-lab/habitat-baselines/habitat_baselines/config/instance_imagenav/ddppo_instance_imagenav.yaml`
- `path/to/habitat-lab/habitat/config/benchmark/nav/instance_imagenav/instance_imagenav_hm3d_v2.yaml`

