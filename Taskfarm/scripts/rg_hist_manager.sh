#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=3988
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --signal=B:TERM@60
#SBATCH --kill-on-invalid-dep=yes
#SBATCH --job-name=rg_hist
#SBATCH --output=../job_outputs/bootstrap/%x_%A.out
#SBATCH --error=../job_logs/bootstrap/%x_%A.err

# Define the constants for this RG flow
TYPE="$1" # Mode of RG workflow
VERSION="$2" # Version for tracking changes and matrix used
N="$3" # Target number of samples
RG_STEP="$4" # The RG step we're currently at
SEED="$5" # Starting seed
RESAMPLE="$6" # Flag to determine which type of resample to use
SYMMETRISE="$7" # Flag to determine whether to symmetrise data or not
SHIFT="${8-}" # Takes in the shift value if running EXP, to change histogram domain

set -euo pipefail

# Directories we're using
basedir="$(cd "$SLURM_SUBMIT_DIR/.."&&pwd)" # Our root directory
codedir="$basedir/code" # Where the code lives

# If we're doing an EXP run, set the directories accordingly
if [[ -n "${SHIFT}" ]]; then
    logsdir="$basedir/job_logs/${VERSION}/$TYPE/shift_${SHIFT}/${SLURM_JOB_NAME}/RG${RG_STEP}" # Where logs will be sent
    outputdir="$basedir/job_outputs/${VERSION}/$TYPE/shift_${SHIFT}" # Where the outputs will live
else
    logsdir="$basedir/job_logs/${VERSION}/$TYPE/${SLURM_JOB_NAME}/RG${RG_STEP}" # Where logs will be sent
    outputdir="$basedir/job_outputs/${VERSION}/$TYPE" # Where the outputs will live
fi

# Common directories regardless of TYPE
joboutdir="$outputdir/output/${SLURM_JOB_NAME}/RG${RG_STEP}" # General output directory
jobdatadir="$outputdir/data" # Where the data will go
batchdir="$jobdatadir/RG${RG_STEP}/batches" # Make a folder for the batches, combined can stay out later
histdir="$jobdatadir/RG${RG_STEP}/hist" # Make a folder for the histograms
statsdir="$jobdatadir/RG${RG_STEP}/stats"
laundereddir="$jobdatadir/RG${RG_STEP}/laundered"


T_DIR="$histdir/t" # Output from rg_gen_batch
Z_DIR="$histdir/z" # Histogram of unsymmetrised z
INPUT_DIR="$histdir/input_t" # Histogram of laundered input t for next RG step
SYM_DIR="$histdir/sym_z" # Histogram of symmetrised z for FP runs

mkdir -p "$T_DIR" "$Z_DIR" "$INPUT_DIR" "$SYM_DIR"
mkdir -p "$outputdir" "$logsdir" # Make these now so that it does it every time we run this job
mkdir -p "$joboutdir" "$jobdatadir" "$batchdir" "$histdir" "$statsdir" "$laundereddir"

exec >"$joboutdir/${SLURM_JOB_NAME}_JOB${SLURM_JOB_ID}.out" # Redirect outputs to be within their own folders, together with the data they produce
exec 2>"$logsdir/${SLURM_JOB_NAME}_JOB${SLURM_JOB_ID}.err" # Redirect error logs to be within their own folders for easy grouping

NUM_BATCHES=$(find "$batchdir" -maxdepth 1 -type d -name "batch_*" | wc -l) # Number of batches of data to generate/process, same as array size

# If no batch folders were detected, the gen job went wrong so just terminate early
if [[ "$NUM_BATCHES" -le 0 ]]; then
    echo "[ERROR]: No batch_* directories found in $batchdir" >&2
    ls -lah "$jobdatadir/RG${RG_STEP}" || true
    exit 1
fi

BATCH_SIZE=$(( N / NUM_BATCHES )) # How many samples exist per batch

# General job information
echo "==================================================="
echo "                  SLURM JOB INFO "
echo "---------------------------------------------------"
echo " Job Name         : $SLURM_JOB_NAME"
echo " Job ID           : $SLURM_JOB_ID"
echo " Submitted from   : $SLURM_SUBMIT_DIR"
echo " Type             : $TYPE"
echo " Shift            : ${SHIFT:-None}"
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
        PREV_Z_HIST="$jobdatadir/RG${PREV_RG}/hist/z/z_hist_unsym_RG${PREV_RG}.npz"
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

# Target filenames for global histogram per RG step
OUTPUT_T="$T_DIR/t_hist_RG${RG_STEP}.npz"
OUTPUT_Z="$Z_DIR/z_hist_unsym_RG${RG_STEP}.npz"

# Local directories to reduce shared I/O load
tempdir="${TMPDIR:-/tmp}/rg_hist_${SLURM_JOB_ID}_RG${RG_STEP}"
tempdir_t="$tempdir/t"
tempdir_z="$tempdir/z"
tempdir_input="$tempdir/input_t"

mkdir -p "$tempdir_t" "$tempdir_z" "$tempdir_input"

temp_output_t="$tempdir_t/t_hist_RG${RG_STEP}.npz"
temp_output_z="$tempdir_z/z_hist_unsym_RG${RG_STEP}.npz"
temp_input_t="$tempdir_input/input_t_hist_RG${RG_STEP}.npz"

echo " Making histograms for RG step $RG_STEP from $NUM_BATCHES batches "
echo " Local histograms stored at: "
echo " t        : $temp_output_t "
echo " z        : $temp_output_z "
echo " input t  : $temp_input_t "

# Histogram jobs for every batch of data
for batch in $(seq 0 $(( NUM_BATCHES - 1 ))); do
    BATCH_DIR="$batchdir/batch_${batch}"

    # Make sure the batch folder exists
    if [[ ! -d "$BATCH_DIR" ]]; then
        echo "[ERROR]: The batch directory is empty: $BATCH_DIR" >&2
        exit 1
    fi

    batch_t="$BATCH_DIR/t_data_RG${RG_STEP}_${BATCH_SIZE}_samples.npy"
    batch_z="$BATCH_DIR/z_data_RG${RG_STEP}_batch_${batch}.npy"

    echo " Adding batch $batch"
    echo " t file : $batch_t"
    echo " z file : $batch_z"

    # Convert generated t' into z
    python -m "source.helpers" \
        3 \
        "$BATCH_SIZE" \
        "$batch_t" \
        "$batch_z" \
        "$RESAMPLE" \
        "$SEED"

    echo " Converted t' data to z data "
    # Construct/Append t' histogram, t has no shift
    if [[ ! -f "$temp_output_t" ]]; then
        python -m "source.histogram_manager" \
            0 \
            "t" \
            "$batch_t" \
            "$temp_output_t" \
            "$RG_STEP"
    else
        python -m "source.histogram_manager" \
            1 \
            "t" \
            "$batch_t" \
            "$temp_output_t" \
            "$temp_output_t" \
            "$RG_STEP"
    fi

    #sleep 1

    # Construct/Append z histogram, with shift
    if [[ ! -f "$temp_output_z" ]]; then
        python -m "source.histogram_manager"\
            0 \
            "z" \
            "$batch_z" \
            "$temp_output_z" \
            "$RG_STEP" \
            "$SHIFT"
    else
        python -m "source.histogram_manager"\
            1 \
            "z" \
            "$batch_z" \
            "$temp_output_z" \
            "$temp_output_z" \
            "$RG_STEP" \
            "$SHIFT"
    fi

    #sleep 5
done
# Check for debugging
ls -lh "$tempdir_t" "$tempdir_z"

# Copy over the histograms from temp storage to shared fs
mv "$temp_output_t" "$OUTPUT_T"
mv "$temp_output_z" "$OUTPUT_Z"

echo " Histograms moved from temp directory to shared FS "

echo " Final histograms at: "
echo " T: $OUTPUT_T "
echo " Z: $OUTPUT_Z "

# Symmetrisation if its an FP run
if [[ "$SYMMETRISE" == "1" ]]; then
    symmetrised_z="$SYM_DIR/sym_z_hist_RG${RG_STEP}.npz"
    python -m "source.helpers" \
        1 \
        "$N" \
        "$OUTPUT_Z" \
        "$symmetrised_z" \
        "$RESAMPLE" \
        "$SEED"

    sampling_hist="$symmetrised_z"
    echo " Symmetrised z histogram saved to: $symmetrised_z "
else
    symmetrised_z="$OUTPUT_Z"
    sampling_hist="$OUTPUT_T"
    echo " Performing RG iters for Type $TYPE "
    echo " Raw z histogram being used at $symmetrised_z"
    echo " Laundering from raw t histogram at $sampling_hist"
fi

# Laundering data for next RG step + histogram job for laundered t

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
            "$launderbatch" \
            "$RESAMPLE" \
            "$SEED"
    else
        python -m "source.helpers" \
            2 \
            "$BATCH_SIZE" \
            "$sampling_hist" \
            "$launderbatch" \
            "$RESAMPLE" \
            "$SEED"
    fi
    #sleep 1
    echo " Batch $batch of t data laundered from $TYPE histogram saved to $launderbatch "
    # Build the input t histogram
    echo " Building histogram for input t data of RG${RG_STEP} "
    if [[ ! -f "$temp_input_t" ]]; then
        python -m "source.t_laundered_hist_manager" \
            0 \
            "$launderbatch" \
            "$temp_input_t" \
            "$RG_STEP"
    else
        python -m "source.t_laundered_hist_manager" \
            1 \
            "$launderbatch" \
            "$temp_input_t" \
            "$temp_input_t" \
            "$RG_STEP"
    fi
    #sleep 5
done
# Check for debugging
ls -lh "$tempdir_input"

# Move back to shared FS
mv "$temp_input_t" "$INPUT_T"

echo " Input t histogram moved from temp storage to shared FS "
echo " Input t histogram for RG${RG_STEP} saved to ${INPUT_T} "

if (( $PREV_RG >= 0 )); then
    echo " [$(date '+%Y-%m-%d %H:%M:%S')] : Clearing data files for RG${PREV_RG} "
    prev_dir="$jobdatadir/RG${PREV_RG}"
    if [[ -d "$prev_dir/batches" ]]; then
        rm -rf "$prev_dir/batches"
    fi
    if [[ -d "$prev_dir/laundered" ]]; then
        rm -rf "$prev_dir/laundered"
    fi
    if [[ "$TYPE" == "EXP" ]]; then
        if [[ -d "$jobdatadir/Initial" ]]; then
            rm -rf "$jobdatadir/Initial"
        fi
        echo " [$(date '+%Y-%m-%d %H:%M:%S')] : Initial shifted files for Shift ${SHIFT} deleted "
    fi
    echo " [$(date '+%Y-%m-%d %H:%M:%S')] : Batch and laundered files for RG${PREV_RG} deleted "
fi

# Clean up temp directory
rm -rf "$tempdir"

echo "=============================================================================================="
echo " Histogram job ${SLURM_JOB_ID} for RG${RG_STEP} completed on : [$(date '+%Y-%m-%d %H:%M:%S')] "
echo "=============================================================================================="
echo ""

wait
sync
exit 0
