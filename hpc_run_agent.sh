#!/usr/bin/env sh
#SBATCH --no-requeue
#SBATCH --partition accelerated
#SBATCH --nodes 1
#SBATCH --gres gpu:4
#SBATCH --mail-user oliver.neumann@kit.edu
#SBATCH --mail-type FAIL
#SBATCH --output=slurm/job-%j.out

module load devel/cuda/12.0
source venv/bin/activate

n_gpus=$(expr $1 - 1)
n_runs_per_agent=$2
sweep_link=$3

for i in $(seq 0 $n_gpus);
do
    export CUDA_VISIBLE_DEVICES=$i
    export WORKER=$i
    wandb agent --count $n_runs_per_agent $sweep_link &
    # wait to avoid runtime errors while accessing data sets
    sleep 60
done

wait
