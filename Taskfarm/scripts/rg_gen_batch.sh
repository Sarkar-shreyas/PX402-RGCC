#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=3988
#SBATCH --cpus-per-task=1
#SBATCH --array=0-31%4
#SBATCH --time=01:00:00
#SBATCH --signal=B:TERM@30
#SBATCH --kill-on-invalid-dep=yes
#SBATCH --job-name=rg_gen
#SBATCH --output=../job_outputs/bootstrap/%x_%A_%a.out
#SBATCH --error=../job_logs/bootstrap/%x_%A_%a.err

set -euo pipefail
STREAM=1 # Additional var for deterministic seeding
UPDATED_CONFIG="$1"
# Define the constants for this RG flow
TYPE="$2" # Type flag to toggle symmetrisation/launder target
VERSION="$3" # Version for tracking changes and matrix used
N="$4" # Target number of samples
RG_STEP="$5" # The RG step we're currently at
SEED="$6" # Starting seed
METHOD="$7" # Flag to determine what method to use
EXPR="$8" # Flag to determine which expression to use
INITIAL="$9" # Flag to generate starting distribution/histograms or not
EXISTING_T_FILE="${10:-}" # Placeholder var to point to data file for non-initial RG steps
SHIFT="${11-}" # Takes in the shift value if running EXP, mostly for folder location
TASK_ID=${SLURM_ARRAY_TASK_ID} # Array task ID for easy tracking
export RG_CONFIG=$UPDATED_CONFIG
# Libraries needed
module purge
module load GCC/13.3.0 SciPy-bundle/2024.05

# Directories we're using
basedir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.."&&pwd)" # Our root directory
codedir="$basedir/code" # Where the code lives
tempdir="${TMPDIR:-/tmp}/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}_${TASK_ID}" # The temp directory we'll use, unique to each job ID
tempbatchdir="$tempdir/RG${RG_STEP}/batch_${TASK_ID}" # The temp directory to write batch data to

# If we're doing an EXP run, set the directories accordingly
if [[ -n "${SHIFT}" ]]; then
    logsdir="$basedir/job_logs/${VERSION}/$TYPE/shift_${SHIFT}/${SLURM_JOB_NAME}/RG${RG_STEP}" # Where logs will be sent
    outputdir="$basedir/job_outputs/${VERSION}/$TYPE/shift_${SHIFT}" # Where the outputs will live
else
    logsdir="$basedir/job_logs/${VERSION}/$TYPE/${SLURM_JOB_NAME}/RG${RG_STEP}" # Where logs will be sent
    outputdir="$basedir/job_outputs/${VERSION}/$TYPE" # Where the outputs will live
fi
# Common directories regardless of TYPE
joboutdir="$outputdir/output/${SLURM_JOB_NAME}/RG${RG_STEP}" # Where the output files will go
jobdatadir="$outputdir/data" # Where the data will live
batchdir="$jobdatadir/RG${RG_STEP}/batches" # Make a folder for the batches, combined can stay out later
batchsubdir="$tempbatchdir"

if [[ "$TYPE" == "FP" ]]; then
    JOB_SEED=$(( SEED + 1000000*RG_STEP + 1000*TASK_ID + STREAM ))
else
    JOB_SEED=$(( SEED + 1000000*RG_STEP + 10000*TASK_ID + STREAM ))
fi
# Make these now so that it does it every time we run this job
mkdir -p "$outputdir" "$logsdir"
mkdir -p "$joboutdir" "$jobdatadir" "$batchdir"
mkdir -p "$tempbatchdir"

out_file="$joboutdir/${SLURM_JOB_NAME}_JOB${SLURM_ARRAY_JOB_ID}_TASK${TASK_ID}.out"
err_file="$logsdir/${SLURM_JOB_NAME}_JOB${SLURM_ARRAY_JOB_ID}_TASK${TASK_ID}.err"
exec >"$out_file" # Redirect outputs to be within their own folders, together with the data they produce
exec 2>"$err_file" # Redirect error logs to be within their own folders for easy grouping

echo "Redirecting output logs to $out_file"
echo "Redirecting error logs to $err_file"

source "$basedir/.venv/bin/activate"

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
echo " Array Job ID     : $SLURM_ARRAY_JOB_ID"
echo " Array Task ID    : $SLURM_ARRAY_TASK_ID"
echo " Submitted from   : $SLURM_SUBMIT_DIR"
echo " Current dir      : $(pwd)"
echo " Date of job      : [$(date '+%Y-%m-%d %H:%M:%S')] "
echo "==================================================="
echo ""

# Print out the config we're at right now (aligned to look nicer :D)

echo "====================================================================="
echo "      Config for data gen of batch no. $TASK_ID for RG step $RG_STEP "
echo "---------------------------------------------------------------------"
echo " RG step           : $RG_STEP"
echo " Total samples     : $N"
echo " Method            : $METHOD "
echo " Expression        : $EXPR "
echo " No. of batches    : $NUM_BATCHES"
echo " Batch size        : $BATCH_SIZE"
echo " Batch directory   : $batchdir"
echo "====================================================================="
echo ""

# Make sure the system recognises the python path to ensure relative imports proceed without issue
export PYTHONPATH="$codedir:$PYTHONPATH"
cd "$codedir"
SRC_DIR="$codedir/source" # This is where the actual code lives
PREV_RG=$(( RG_STEP - 1 ))

# Handle what data to use
if [[ "$INITIAL" -eq 0 ]]; then
    if [[ -d "$EXISTING_T_FILE" ]]; then
        if [[ "$TYPE" == "FP" ]]; then
            # If its the first step of the FP run, use t_laundered as usual
            T_INPUT="$EXISTING_T_FILE/t_laundered_RG${PREV_RG}_batch_${TASK_ID}.npy"
        else
            if [[ "$RG_STEP" -eq 0 ]]; then
                # If its the first step of the EXP run, use the shifted dataset
                T_INPUT="$EXISTING_T_FILE/perturbed_t_shift_${SHIFT}_batch_${TASK_ID}.npy"
            else
                # Otherwise, use the usual laundered dataset
                T_INPUT="$EXISTING_T_FILE/t_laundered_RG${PREV_RG}_batch_${TASK_ID}.npy"
            fi
        fi
    else
        # If its just a file and not a directory, we can use it directly
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
        "$JOB_SEED" \
        "$T_INPUT"
else
    python -m "source.data_generation" \
    "$BATCH_SIZE" \
    "$batchsubdir" \
    "$INITIAL" \
    "$RG_STEP" \
    "$JOB_SEED"
fi

# Move batch back to shared storage
target_dir="$batchdir/batch_${TASK_ID}"
mkdir -p "$target_dir"
if timeout 45 rsync -a --partial "$tempbatchdir/" "$target_dir/"; then
    rm -rf "$tempbatchdir"
    echo " Data from $tempbatchdir deleted and moved to $target_dir "
    echo "OK" > "$target_dir/READY"
else
    echo " [Warning]: rsync failed for batch ${TASK_ID}, leaving tmp at $tempbatchdir" >&2
    exit 1
fi

# Free the tmp folder
echo "==================================================================================================="
echo " Data gen job ${SLURM_ARRAY_JOB_ID} for RG${RG_STEP} completed on : [$(date '+%Y-%m-%d %H:%M:%S')] "
echo "==================================================================================================="
echo ""

exit 0
