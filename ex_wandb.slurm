#!/bin/bash
#SBATCH --time=0-16:00:00 #day-h:m:s
#SBATCH --job-nam=UW_C #Job name
#SBATCH --partition=all #gpu partition
#SBATCH --mem=64G #ram alocation
#SBATCH -o /mundus/mdeowan698/ppo_result.log #parmeter log
#SBATCH -e /mundus/mdeowan698/ppo_error.log #error log

#SBATCH --gres=gpu:a40-48:1
#SBATCH --partition=all # any gpu

cd /mundus/mdeowan698/BlueROV_Navigation-Control
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ROV

CUDA_LAUNCH_BLOCKING=1 python3 trpo_gym.py --save-model-interval 5 --env-name navigation --eval-batch-size 0 --min-batch-size 2048 --num-threads 1 --hist-length 5 --gpu-index 0 --adaptation 0 --max-iter-num 200
#CUDA_LAUNCH_BLOCKING=1 python3 ppo_gym_test.py --env-name navigation --eval-batch-size 2000 --hist-length 5
# gpu:a100-20:1

