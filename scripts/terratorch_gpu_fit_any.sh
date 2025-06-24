#!/bin/sh
#SBATCH -A ka1176
#SBATCH -p gpu
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=10
#SBATCH --constraint=a100_80
#SBATCH -o scripts/slurm/slurm-%j.out

source ~/.bashrc
conda activate terramind

# pass e.g. config/arctic_sea_ice/sic.yaml as the first argument

terratorch fit -c $1 --custom_modules_path="/work/ka1176/caroline/gitlab/terramind-demo" 

wandb artifact cache cleanup --remove-temp 0GB