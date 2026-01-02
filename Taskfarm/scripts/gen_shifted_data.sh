#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=3988
#SBATCH --cpus-per-task=1
#SBATCH --array=0-31%4
#SBATCH --time=01:00:00
#SBATCH --signal=B:TERM@30
#SBATCH --kill-on-invalid-dep=yes
#SBATCH --job-name=gen_shift
#SBATCH --output=../job_outputs/bootstrap/%x_%A_%a.out
#SBATCH --error=../job_logs/bootstrap/%x_%A_%a.err
set -euo pipefail
UPDATED_CONFIG="$1"
STREAM=4 # Additional var for deterministic seeding
# Define the constants for this RG flow
VERSION="$2" # Version for tracking changes and matrix used
N="$3" # Total number of samples
INPUT_FILE="$4" # The input histogram we launder and shift from
STEP="$5" # The RG step we're currently at
SEED="$6" # Starting seed
SHIFT="$7" # Takes in the shift value to apply for this round.
TYPE="EXP" # Type flag to toggle symmetrisation/launder target
TASK_ID=${SLURM_ARRAY_TASK_ID} # Array task ID for easy tracking
export RG_CONFIG=$UPDATED_CONFIG
# Libraries needed
module purge
module load GCC/13.3.0 SciPy-bundle/2024.05

basedir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.."&&pwd)" # Our root directory
codedir="$basedir/code" # Where the code lives
logsdir="$basedir/job_logs/${VERSION}/$TYPE/shift_${SHIFT}/${SLURM_JOB_NAME}" # Where log files will go
outputdir="$basedir/job_outputs/${VERSION}/$TYPE/shift_${SHIFT}" # General output dir for this shift
joboutdir="$outputdir/output/${SLURM_JOB_NAME}/${STEP}" # Where the output files will go
jobdatadir="$outputdir/data" # Where the data will go
stepdir="$jobdatadir/${STEP}" # The data directory for this RG step
mkdir -p "$outputdir" "$logsdir" # Make them in case they aren't already there
mkdir -p "$joboutdir" "$jobdatadir" "$stepdir" # Make them in case they aren't already there

out_file="$joboutdir/${SLURM_JOB_NAME}_JOB${SLURM_ARRAY_JOB_ID}_TASK${TASK_ID}.out"
err_file="$logsdir/${SLURM_JOB_NAME}_JOB${SLURM_ARRAY_JOB_ID}_TASK${TASK_ID}.err"
exec >"$out_file" # Redirect outputs to be within their own folders, together with the data they produce
exec 2>"$err_file" # Redirect error logs to be within their own folders for easy grouping

echo "Redirecting output logs to $out_file"
echo "Redirecting error logs to $err_file"

source "$basedir/.venv/bin/activate"
JOB_SEED=$(( SEED + 123456 + 1000*TASK_ID + STREAM ))

NUM_BATCHES=$(( SLURM_ARRAY_TASK_MAX + 1 )) # Number of batches to generate/process data over, same as array size
if (( N % NUM_BATCHES != 0 )); then
    echo "[ERROR] N=$N not divisible by NUM_BATCHES=$NUM_BATCHES" >&2
    exit 2
fi
BATCH_SIZE=$(( N / NUM_BATCHES )) # How many samples should be calculated per batch

# General job information
echo "==================================================="
echo "                  SLURM JOB INFO "
echo "---------------------------------------------------"
echo " Job Name         : $SLURM_JOB_NAME"
echo " Job ID           : $SLURM_JOB_ID"
echo " Submitted from   : $SLURM_SUBMIT_DIR"
echo " Current dir      : $(pwd)"
echo " Date of job      : [$(date '+%Y-%m-%d %H:%M:%S')] "
echo "==================================================="
echo ""

# Config for confirmation purposes
echo "====================================================================="
echo "      Config for shifting samples from the FP for shift $SHIFT "
echo "---------------------------------------------------------------------"
echo " Total samples         : $N"
echo " Input FP distribution : $INPUT_FILE"
echo " Output data directory : $stepdir"
echo "====================================================================="
echo ""

# Make sure the system recognises the python path to ensure relative imports proceed without issue
export PYTHONPATH="$codedir:$PYTHONPATH"
cd "$codedir"
SRC_DIR="$codedir/source" # This is where the actual code lives

# Where to store the shifted data for use in later RG steps
OUTPUT_FILE="$jobdatadir/${STEP}/perturbed_t_shift_${SHIFT}_batch_${TASK_ID}.npy"

python -m "source.shift_z" \
    "$BATCH_SIZE" \
    "$INPUT_FILE" \
    "$OUTPUT_FILE" \
    "$JOB_SEED" \
    "$SHIFT"

echo "======================================================================================================="
echo " Data shift job ${SLURM_JOB_ID} for Shift ${SHIFT} completed on : [$(date '+%Y-%m-%d %H:%M:%S')] "
echo "======================================================================================================="
echo ""

exit 0
