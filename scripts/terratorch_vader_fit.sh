#!/bin/sh
#SBATCH -A ka1176
#SBATCH -p vader
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH -o scripts/slurm/slurm-%j.out

source ~/.bashrc
conda activate terramind

terratorch fit -c config/arctic_sea_ice/sic.yaml --custom_modules_path="/work/ka1176/caroline/gitlab/terramind-demo"