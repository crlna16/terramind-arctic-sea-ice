#!/bin/sh
#SBATCH -A ka1176
#SBATCH -p vader
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --ntasks=10
#SBATCH -o scripts/slurm/slurm-%j.out

source ~/.bashrc
conda activate terramind

terratorch fit -c config/arctic_sea_ice/sic.yaml --custom_modules_path="/work/ka1176/caroline/gitlab/terramind-demo" 

wandb artifact cache cleanup --remove-temp 0GB