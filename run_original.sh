#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=3988
#SBATCH --cpus-per-task=1
#SBATCH --time=08:00:00
#SBATCH --job-name=test_main
#SBATCH --output=../job_outputs/bootstrap/%x_%A_%a.out
#SBATCH --error=../job_logs/bootstrap/%x_%A_%a.err

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

module purge
module load GCC/13.3.0 SciPy-bundle/2024.05 matplotlib/3.9.2

basedir="$(cd "$SLURM_SUBMIT_DIR/.."&&pwd)"
codedir="$basedir/code"
jobsdir="$basedir/jobs"
logsdir="$basedir/job_logs"
outputdir="$basedir/job_outputs"

jobkey="${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}"
task="${SLURM_ARRAY_TASK_ID:-0}"
joblogdir="$logsdir/$jobkey"
joboutdir="$outputdir/$jobkey"
mkdir -p "$joblogdir" "$joboutdir" "$joboutdir/output" "$joboutdir/data"

exec > >(tee -a "$joboutdir/output/task_${task}.out")
exec 2> >(tee -a "$joblogdir/task_${task}.err" >&2)

N=30000000
K=8

echo "Task $SLURM_ARRAY_TASK_ID -> N=$N K=$K output=$joboutdir"

cd "$codedir"
export PYTHONPATH="$codedir:$PYTHONPATH"
python test_main.py "$N" "$K" "$joboutdir"
