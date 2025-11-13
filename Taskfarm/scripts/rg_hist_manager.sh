#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=3988
#SBATCH --cpus-per-task=1
#SBATCH --time=08:00:00
#SBATCH --job-name=rg_hist
#SBATCH --output=../job_outputs/bootstrap/%x_%A.out
#SBATCH --error=../job_logs/bootstrap/%x_%A.err

# Define the constants for this RG flow
VERSION="$4" # Version for tracking changes and matrix used
N="$1" # Target number of samples
RG_STEP="$2" # The RG step we're currently at
SHIFT="${5-}" # Takes in the shift value if running EXP, mostly for folder location
TYPE="$3" # Type flag to toggle symmetrisation/launder target
NUM_BATCHES=8 # Number of batches of data to generate/process, same as array size
BATCH_SIZE=$(( N / NUM_BATCHES )) # How many samples exist per batch
set -euo pipefail

# Directories we're using
basedir="$(cd "$SLURM_SUBMIT_DIR/.."&&pwd)" # Our root directory
codedir="$basedir/code" # Where the code lives

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
joboutdir="$outputdir/output/${SLURM_JOB_NAME}/RG${RG_STEP}" # General output directory
jobdatadir="$outputdir/data" # Where the data will go
batchdir="$jobdatadir/RG${RG_STEP}/batches" # Make a folder for the batches, combined can stay out later
histdir="$jobdatadir/RG${RG_STEP}/hist" # Make a folder for the histograms
statsdir="$jobdatadir/RG${RG_STEP}/stats"
laundereddir="$jobdatadir/RG${RG_STEP}/laundered"


T_DIR="$histdir/t"
G_DIR="$histdir/g"
Z_DIR="$histdir/z"
INPUT_DIR="$histdir/input_t"
SYM_DIR="$histdir/sym_z"

mkdir -p "$T_DIR" "$G_DIR" "$Z_DIR" "$INPUT_DIR" "$SYM_DIR"
mkdir -p "$outputdir" "$logsdir" "$jobsdir" # Make these now so that it does it every time we run this job
mkdir -p "$joboutdir" "$jobdatadir" "$batchdir" "$histdir" "$statsdir" "$laundereddir"

exec > >(tee -a "$joboutdir/RG_${RG_STEP}_JOB${SLURM_JOB_ID}.out") # Redirect outputs to be within their own folders, together with the data they produce
exec 2> >(tee -a "$logsdir/RG_${RG_STEP}_JOB${SLURM_JOB_ID}.err" >&2) # Redirect error logs to be within their own folders for easy grouping

# General job information
echo "==================================================="
echo "                  SLURM JOB INFO "
echo "---------------------------------------------------"
echo " Job Name         : $SLURM_JOB_NAME"
echo " Job ID           : $SLURM_JOB_ID"
echo " Submitted from   : $SLURM_SUBMIT_DIR"
echo " Type             : $TYPE"
echo " Current dir      : $(pwd)"
echo " Date of job      : [$(date '+%Y-%m-%d %H:%M:%S')] "
echo "==================================================="
echo ""


# Store prev iter number for moments later
PREV_RG=$(( RG_STEP - 1 ))
if [[ $RG_STEP -eq 0 ]]; then
    PREV_Z_HIST=""
else
    if [[ "$TYPE" == "FP" ]]; then
        PREV_Z_HIST="$jobdatadir/RG${PREV_RG}/hist/sym_z/sym_z_hist_RG${PREV_RG}.npz"
    else
        PREV_Z_HIST="$jobdatadir/RG${PREV_RG}/hist/z/z_hist_RG${PREV_RG}.npz"
    fi
fi

# Print out the config we're at right now (aligned to look nicer :D)
echo "===================================================="
echo "      Config for hist gen of RG step $RG_STEP "
echo "----------------------------------------------------"
echo " RG step               : $RG_STEP"
echo " Total samples         : $N"
echo " No. of batches        : $NUM_BATCHES"
echo " Batch size            : $BATCH_SIZE"
echo " Batch directory       : $batchdir"
echo " Hist directory        : $histdir"
echo " Stats directory       : $statsdir"
echo " Laundered t directory : $laundereddir"
echo " Previous z hist       : ${PREV_Z_HIST:-None}"
echo "===================================================="
echo ""

# Libraries needed
module purge
module load GCC/13.3.0 SciPy-bundle/2024.05

# Make sure the system recognises the python path to ensure relative imports proceed without issue
export PYTHONPATH="$codedir:$PYTHONPATH"
cd "$codedir"
SRC_DIR="$codedir/source" # This is where the actual code lives

# Target filenames for global histogram per RG step
OUTPUT_T="$T_DIR/t_hist_RG${RG_STEP}.npz"
OUTPUT_G="$G_DIR/g_hist_RG${RG_STEP}.npz"
OUTPUT_Z="$Z_DIR/z_hist_RG${RG_STEP}.npz"

echo " Making histograms for RG step $RG_STEP from $NUM_BATCHES batches "

# Process every batch of data
for batch in $(seq 0 $(( NUM_BATCHES - 1 ))); do
    BATCH_DIR="$batchdir/batch_${batch}"

    batch_t=$(ls "$BATCH_DIR"/t_data_RG${RG_STEP}_*.npy)
    batch_g=$(ls "$BATCH_DIR"/g_data_RG${RG_STEP}_*.npy)
    batch_z=$(ls "$BATCH_DIR"/z_data_RG${RG_STEP}_*.npy)

    echo " Adding batch $batch:"
    echo " t dir: $batch_t"
    echo " g dir: $batch_g"
    echo " z dir: $batch_z"

    # 2 choices depending on whether its the first histogram or not
    if [[ ! -f "$OUTPUT_T" || ! -f "$OUTPUT_G" || ! -f "$OUTPUT_Z" ]]; then
        python -m "source.histogram_manager" \
            0 \
            "$batch_t" \
            "$batch_g" \
            "$batch_z" \
            "$OUTPUT_T" \
            "$OUTPUT_G" \
            "$OUTPUT_Z" \
            "$RG_STEP"
    else
        python -m "source.histogram_manager" \
            1 \
            "$batch_t" \
            "$batch_g" \
            "$batch_z" \
            "$OUTPUT_T" \
            "$OUTPUT_G" \
            "$OUTPUT_Z" \
            "$OUTPUT_T" \
            "$OUTPUT_G" \
            "$OUTPUT_Z" \
            "$RG_STEP"
    fi
    sleep 5
done

echo " Made histograms at: "
echo " T: $OUTPUT_T "
echo " G: $OUTPUT_G "
echo " Z: $OUTPUT_Z "

# If we're doing the FP calculation, we symmetrise, otherwise we don't.
# For FP, we launder form Sym Z, for EXP, we launder from t
if [[ "$TYPE" == "FP" ]]; then
    symmetrised_z="$SYM_DIR/sym_z_hist_RG${RG_STEP}.npz"
    python -m "source.helpers" \
        1 \
        "$N" \
        "$OUTPUT_Z" \
        "$symmetrised_z"
    sampling_hist="$symmetrised_z"
    echo " Symmetrised z histogram saved to: $symmetrised_z "
else
    symmetrised_z="$OUTPUT_Z"
    sampling_hist="$OUTPUT_T"
    echo " Performing RG iters for Type $TYPE "
    echo " Raw z histogram being used at $symmetrised_z"
    echo " Laundering from raw t histogram at $sampling_hist"
fi

# Analyse the moments

stats="$statsdir/z_moments_RG${RG_STEP}_${N}_samples.npz"

if [[ -z "$PREV_Z_HIST" || ! -f "$PREV_Z_HIST" ]]; then
    prev="None"
else
    prev="$PREV_Z_HIST"
fi
# If we're doing the FP we can calculate these, for EXP thats done locally while doing peak finding
if [[ "$TYPE" == "FP" ]]; then
    python -m "source.rg" \
        "$RG_STEP" \
        "$prev" \
        "$symmetrised_z" \
        "$stats"

    echo " Moments of histogram for $N samples written to $stats. "
fi
# Point to the input t histogram so we can construct it
INPUT_T="$INPUT_DIR/input_t_hist_RG${RG_STEP}.npz"

for batch in $(seq 0 $(( NUM_BATCHES - 1 ))); do
    launderbatch="$laundereddir/t_laundered_RG${RG_STEP}_batch_${batch}.npy"
    # Produce batches of laundered data, from z if TYPE=FP and from t otherwise
    if [[ "$TYPE" == "FP" ]]; then
        python -m "source.helpers" \
            0 \
            "$BATCH_SIZE" \
            "$sampling_hist" \
            "$launderbatch"
    else
        python -m "source.helpers" \
            2 \
            "$BATCH_SIZE" \
            "$sampling_hist" \
            "$launderbatch"
    fi
    sleep 5
    echo " Batch $batch of t data laundered from Q(z) saved to $launderbatch "
    # Build the input t histogram
    echo " Building histogram for input t data of RG${RG_STEP} "
    if [[ ! -f "$INPUT_T" ]]; then
        python -m "source.t_laundered_hist_manager" \
            0 \
            "$launderbatch" \
            "$INPUT_T" \
            "$RG_STEP"
    else
        python -m "source.t_laundered_hist_manager" \
            1 \
            "$launderbatch" \
            "$INPUT_T" \
            "$INPUT_T" \
            "$RG_STEP"
    fi
    sleep 5
done

echo " Input t histogram for RG${RG_STEP} built at ${INPUT_T} "

echo "=============================================================================================="
echo " Histogram job ${SLURM_JOB_ID} for RG${RG_STEP} completed on : [$(date '+%Y-%m-%d %H:%M:%S')] "
echo "=============================================================================================="
echo ""


