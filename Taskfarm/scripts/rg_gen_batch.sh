#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=3988
#SBATCH --cpus-per-task=1
#SBATCH --array=0-31%8
#SBATCH --time=08:00:00
#SBATCH --job-name=rg_gen
#SBATCH --output=../job_outputs/bootstrap/%x_%A_%a.out
#SBATCH --error=../job_logs/bootstrap/%x_%A_%a.err

# Define the constants for this RG flow
VERSION="$5" # Version for tracking changes and matrix used
TYPE="$6" # Type flag to toggle symmetrisation/launder target
SHIFT="${7-}" # Takes in the shift value if running EXP, mostly for folder location
N="$1" # Target number of samples
RG_STEP="$4" # The RG step we're currently at
INITIAL="$2" # Flag to generate starting distribution/histograms or not
NUM_BATCHES=$((SLURM_ARRAY_TASK_MAX + 1)) # Number of batches to generate/process data over, same as array size
BATCH_SIZE=$(( N / NUM_BATCHES )) # How many samples should be calculated per batch
TASK_ID=${SLURM_ARRAY_TASK_ID} # Array task ID for easy tracking

EXISTING_T_FILE="$3" # Placeholder var to point to data file for non-initial RG steps
set -euo pipefail

# Libraries needed
module purge
module load GCC/13.3.0 SciPy-bundle/2024.05

# Directories we're using
basedir="$(cd "$SLURM_SUBMIT_DIR/.."&&pwd)" # Our root directory
codedir="$basedir/code" # Where the code lives
tempdir="${TMPDIR:-/tmp}/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}" # The temp directory we'll use, unique to each job ID
tempbatchdir="$tempdir/RG${RG_STEP}/batch_${TASK_ID}" # The temp directory to write batch data to

# If we're doing an EXP run, set the directories accordingly
if [[ -n "${SHIFT}" ]]; then
    jobsdir="$basedir/jobs/v${VERSION}/$TYPE/shift_${SHIFT}" # Where metadata will be
    logsdir="$basedir/job_logs/v${VERSION}/$TYPE/shift_${SHIFT}/${SLURM_JOB_NAME}/RG${RG_STEP}" # Where logs will be sent
    outputdir="$basedir/job_outputs/v${VERSION}/$TYPE/shift_${SHIFT}" # Where the outputs will live
else
    jobsdir="$basedir/jobs/v${VERSION}/$TYPE" # Where metadata will be
    logsdir="$basedir/job_logs/v${VERSION}/$TYPE/${SLURM_JOB_NAME}/RG${RG_STEP}" # Where logs will be sent
    outputdir="$basedir/job_outputs/v${VERSION}/$TYPE" # Where the outputs will live
fi
# Common directories regardless of TYPE
joboutdir="$outputdir/output/${SLURM_JOB_NAME}/RG${RG_STEP}" # Where the output files will go
jobdatadir="$outputdir/data" # Where the data will live
batchdir="$jobdatadir/RG${RG_STEP}/batches" # Make a folder for the batches, combined can stay out later
batchsubdir="$tempbatchdir"


# Make these now so that it does it every time we run this job
mkdir -p "$outputdir" "$logsdir" "$jobsdir"
mkdir -p "$joboutdir" "$jobdatadir" "$batchdir"
mkdir -p "$tempbatchdir"
exec >"$joboutdir/${SLURM_JOB_NAME}_JOB${SLURM_ARRAY_JOB_ID}_TASK${TASK_ID}.out" # Redirect outputs to be within their own folders, together with the data they produce
exec 2>"$logsdir/${SLURM_JOB_NAME}_JOB${SLURM_ARRAY_JOB_ID}_TASK${TASK_ID}.err" # Redirect error logs to be within their own folders for easy grouping

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
        "$T_INPUT"
else
    python -m "source.data_generation" \
    "$BATCH_SIZE" \
    "$batchsubdir" \
    "$INITIAL" \
    "$RG_STEP"
fi

# Move batch back to shared storage
target_dir="$batchdir/batch_${TASK_ID}"
mkdir -p "$target_dir"
rsync -a "$tempbatchdir/" "$target_dir/"

# Free the tmp folder
rm -rf "$tempbatchdir"
echo " Data from $tempbatchdir deleted and moved to $target_dir "

echo "==================================================================================================="
echo " Data gen job ${SLURM_ARRAY_JOB_ID} for RG${RG_STEP} completed on : [$(date '+%Y-%m-%d %H:%M:%S')] "
echo "==================================================================================================="
echo ""
