#!/bin/sh
#SBATCH -A ka1176
#SBATCH -p gpu
#SBATCH --exclusive
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --constraint=a100_80
#SBATCH -o scripts/slurm/slurm-%j.out

source ~/.bashrc
conda activate terramind

# pass e.g. config/arctic_sea_ice/sic.yaml as the first argument

export WANDB_CACHE_DIR="/scratch/k/k202141/wandb_cache"
export WANDB_DATA_DIR ="/scratch/k/k202141/wandb_data"

terratorch fit -c $1 --custom_modules_path="/work/ka1176/caroline/gitlab/terramind-demo"

wandb artifact cache cleanup --remove-temp 0GB