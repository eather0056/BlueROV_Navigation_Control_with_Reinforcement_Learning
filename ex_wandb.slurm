#!/bin/bash
#SBATCH --time=2-00:00:00 #day-h:m:s
#SBATCH --job-nam=UW_C #Job name
#SBATCH --partition=all #gpu partition
#SBATCH --mem=64G #ram alocation
#SBATCH -o /mundus/mdeowan698/UW_C_R_wandb.log #parmeter log
#SBATCH -e /mundus/mdeowan698/UW_C_E_wandb.log #error log

#SBATCH --gres=gpu:a100-20:1
#SBATCH --partition=all # any gpu

cd /mundus/mdeowan698/BlueROV_Navigation-Control
source ~/miniconda3/etc/profile.d/conda.sh
conda activate BR

CUDA_LAUNCH_BLOCKING=1 python3 ppo_gym.py --save-model-interval 5 --env-name navigation --eval-batch-size 0 --min-batch-size 2048 --num-threads 1 --hist-length 5 --gpu-index 0 --adaptation 0 --max-iter-num 200
#CUDA_LAUNCH_BLOCKING=1 python3 ppo_gym_test.py --env-name navigation --eval-batch-size 2000 --hist-length 5