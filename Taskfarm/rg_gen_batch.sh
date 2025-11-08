#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=3988
#SBATCH --cpus-per-task=1
#SBATCH --array=0-7
#SBATCH --time=08:00:00
#SBATCH --job-name=rg_gen
#SBATCH --output=../job_outputs/bootstrap/%x_%A_%a.out
#SBATCH --error=../job_logs/bootstrap/%x_%A_%a.err

# Config variables
VERSION=1.21 # A version =number to help me track where we're at
N="$1" # Target number of samples
RG_STEP="$4" # Step counter
INITIAL="$2" # This is the first run, need initial distribution
NUM_BATCHES=$((SLURM_ARRAY_TASK_MAX + 1)) # Number of batches to split this into, same as array size
BATCH_SIZE=$(( N / NUM_BATCHES ))
TASK_ID=${SLURM_ARRAY_TASK_ID}
# Placeholder for later steps
EXISTING_T_FILE="$3"
set -euo pipefail

# Libraries needed
module purge
module load GCC/13.3.0 SciPy-bundle/2024.05

# Directories we're using
basedir="$(cd "$SLURM_SUBMIT_DIR/.."&&pwd)" # Root, fyp for now
codedir="$basedir/code" # Where the code lives
jobsdir="$basedir/jobs/v${VERSION}" # Where metadata will be
logsdir="$basedir/job_logs/v${VERSION}/${SLURM_JOB_NAME}" # Where logs will be sent
outputdir="$basedir/job_outputs/v${VERSION}/${SLURM_JOB_NAME}" # Where the outputs will live
joboutdir="$outputdir/output"
jobdatadir="$outputdir/data"
batchdir="$jobdatadir/RG${RG_STEP}/batches" # Make a folder for the batches, combined can stay out later
batchsubdir="$batchdir/batch_${TASK_ID}"
mkdir -p "$outputdir" "$logsdir" "$jobsdir" # Make these now so that it does it every time we run this job
mkdir -p "$joboutdir" "$jobdatadir" "$batchdir" "$batchsubdir"

exec > >(tee -a "$joboutdir/RG_${RG_STEP}_JOB${SLURM_JOB_ID}.out")
exec 2> >(tee -a "$logsdir/RG_${RG_STEP}_JOB${SLURM_JOB_ID}.err" >&2)

echo "==================================================="
echo "                  SLURM JOB INFO "
echo "---------------------------------------------------"
echo " Job Name         : $SLURM_JOB_NAME"
echo " Job ID           : $SLURM_JOB_ID"
echo " Array Task ID    : ${SLURM_ARRAY_TASK_ID:-N/A}"
echo " Submitted from   : $SLURM_SUBMIT_DIR"
echo " Current dir      : $(pwd)"
echo " Date of job      : [$(date '+%H:%M:%S')]"
echo "=================================================="
echo ""

# Print out the config we're at right now (aligned to look nicer :D)

echo "==================================================="
echo "      Config for data gen of RG step $RG_STEP "
echo "---------------------------------------------------"
echo "RG step           : $RG_STEP"
echo "Total samples     : $N"
echo "No. of batches    : $NUM_BATCHES"
echo "Batch size        : $BATCH_SIZE"
echo "Batch directory   : $batchdir"
echo "==================================================="
echo ""

# cd to code directory for paths
export PYTHONPATH="$codedir:$PYTHONPATH"
cd "$codedir"
SRC_DIR="$codedir/source" # This is where the actual code lives
PREV_RG=$(( RG_STEP - 1 ))

# If we passed in a directory, it'll run through each file.
if [[ "$INITIAL" -eq 0 ]]; then
    if [[ -d "$EXISTING_T_FILE" ]]; then
        T_INPUT="$EXISTING_T_FILE/t_laundered_RG${PREV_RG}_batch_${TASK_ID}.txt"
    else
        T_INPUT="$EXISTING_T_FILE"
    fi
else
    T_INPUT=""
fi

# Check for existing t file
if [[ -n "$T_INPUT" ]]; then

    python -m "source.data_generation" \
        "$BATCH_SIZE" \
        "$batchsubdir" \
        "$INITIAL" \
        "$RG_STEP" \
        "$T_INPUT"
else
    python -m "source.data_generation" \
    "$BATCH_SIZE" \
    "$batchsubdir" \
    "$INITIAL" \
    "$RG_STEP"
fi

echo "==================================================================================="
echo "Data gen job ${SLURM_JOB_ID} for RG${RG_STEP} completed on : [$(date '+%H:%M:%S')]"
echo "==================================================================================="
echo ""
