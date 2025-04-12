#!/usr/bin/env sh

n_gpus=4
n_runs_per_agent=1

for i in $(seq 1 $1)
do
    echo "Job $i: Memory -> $2 MB Duration -> $3 minutes"
    sbatch --mem $2 --time $3 hpc_run_agent.sh $n_gpus $n_runs_per_agent $4
done
