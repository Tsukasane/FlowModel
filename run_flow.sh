#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH -t 2-00:00:00

# export HF_HOME=/ocean/projects/cis240055p/yzhao16/hub_model
# export DF_CACHE_DIR=/ocean/projects/cis240055p/yzhao16/hub_model
export CUDA_HOME=/ocean/projects/cis210027p/yzhao16/miniconda3/envs/speechlm2
source /ocean/projects/cis210027p/yzhao16/miniconda3/bin/activate speechlm2

python flow.py