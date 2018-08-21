#!/bin/sh
#SBATCH --partition=ai
#SBATCH --gres=gpu:1
#SBATCH --time=56:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=100GB
#SBATCH --job-name="clevr"
#SBATCH --output=clevr.out
#SBATCH --error=clevr.error

julia train.jl src/macnet.jl configs/config2.jl
