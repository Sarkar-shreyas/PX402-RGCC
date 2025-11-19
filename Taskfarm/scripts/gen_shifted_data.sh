#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=3988
#SBATCH --cpus-per-task=1
#SBATCH --array=0-15%4
#SBATCH --time=08:00:00
#SBATCH --job-name=gen_shift
#SBATCH --output=../job_outputs/bootstrap/%x_%A.out
#SBATCH --error=../job_logs/bootstrap/%x_%A.err

# Define the constants for this RG flow
N="$1" # Total number of samples
VERSION="$2" # Version for tracking changes and matrix used
INPUT_FILE="$3" # The input histogram we launder and shift from
STEP="$4" # The RG step we're currently at
shift="$5" # Takes in the shift value to apply for this round.
TYPE="EXP" # Type flag to toggle symmetrisation/launder target
NUM_BATCHES=$((SLURM_ARRAY_TASK_MAX + 1)) # Number of batches to generate/process data over, same as array size
BATCH_SIZE=$(( N / NUM_BATCHES )) # How many samples should be calculated per batch
TASK_ID=${SLURM_ARRAY_TASK_ID} # Array task ID for easy tracking

set -euo pipefail

# Libraries needed
module purge
module load GCC/13.3.0 SciPy-bundle/2024.05

basedir="$(cd "$SLURM_SUBMIT_DIR/.."&&pwd)" # Our root directory
codedir="$basedir/code" # Where the code lives
logsdir="$basedir/job_logs/v${VERSION}/$TYPE/shift_${shift}/${SLURM_JOB_NAME}" # Where log files will go
outputdir="$basedir/job_outputs/v${VERSION}/$TYPE/shift_${shift}" # General output dir for this shift
joboutdir="$outputdir/output/${SLURM_JOB_NAME}/${STEP}" # Where the output files will go
jobdatadir="$outputdir/data" # Where the data will go
stepdir="$jobdatadir/${STEP}" # The data directory for this RG step
mkdir -p "$outputdir" "$logsdir" # Make them in case they aren't already there
mkdir -p "$joboutdir" "$jobdatadir" "$stepdir" # Make them in case they aren't already there

exec >"$joboutdir/${SLURM_JOB_NAME}_JOB${SLURM_JOB_ID}.out" # Redirect outputs to be within their own folders, together with the data they produce
exec 2>"$logsdir/${SLURM_JOB_NAME}_JOB${SLURM_JOB_ID}.err" # Redirect error logs to be within their own folders for easy grouping

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
echo "      Config for shifting samples from the FP for shift $shift "
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
OUTPUT_FILE="$jobdatadir/${STEP}/perturbed_t_shift_${shift}_batch_${TASK_ID}.npy"

python -m "source.shift_z" \
    "$BATCH_SIZE" \
    "$INPUT_FILE" \
    "$OUTPUT_FILE" \
    "$shift"

echo "======================================================================================================="
echo " Data shift job ${SLURM_JOB_ID} for Shift ${shift} completed on : [$(date '+%Y-%m-%d %H:%M:%S')] "
echo "======================================================================================================="
echo ""
