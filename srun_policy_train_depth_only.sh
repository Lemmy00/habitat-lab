#!/bin/bash
#SBATCH -G 4
#SBATCH -c 48                        
#SBATCH --time=36:00:00
#SBATCH --mem-per-cpu=4000
#SBATCH --gpus=titan_rtx:4
#SBATCH -J img_nav_policy_depth_only
#SBATCH -o train_policy_logs/img_nav_policy_depth_only.out
#SBATCH -e train_policy_logs/img_nav_policy_depth_only.err

# Change directory to the submission directory
cd "$SLURM_SUBMIT_DIR" || { echo "Submission directory not found! Exiting."; exit 1; }

# Initialize conda and activate your environment (adjust paths as needed)
unset PYTHONHOME PYTHONPATH && source /path/to/miniconda3/bin/activate habitat

# Set env variables
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

# Run your training command or script
set -x
python -m torch.distributed.launch --nproc_per_node=4 --use_env rl-distance-train/train_image_nav.py --depth-only --dist-to-goal
