#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=3988
#SBATCH --cpus-per-task=1
#SBATCH --time=08:00:00
#SBATCH --job-name=rg_hist_manager
#SBATCH --output=../job_outputs/bootstrap/%x_%A_%a.out
#SBATCH --error=../job_logs/bootstrap/%x_%A_%a.err

# Config variables
VERSION=1 # A version number to help me track where we're at
N="$1" # Target number of samples
RG_STEP="$2" # Step counter
NUM_BATCHES=10 # Number of batches to split this into, same as array size
BATCH_SIZE=$(( N / NUM_BATCHES ))

set -euo pipefail


# Directories we're using
basedir="$(cd "$SLURM_SUBMIT_DIR/.."&&pwd)" # Root, fyp for now
codedir="$basedir/code" # Where the code lives
jobsdir="$basedir/jobs/v${VERSION}" # Where metadata will be
logsdir="$basedir/job_logs/v${VERSION}" # Where logs will be sent
outputdir="$basedir/job_outputs/v${VERSION}" # Where the outputs will live
joboutdir="$outputdir/output" # General output directory
jobdatadir="$outputdir/data" # Where the data will go
batchdir="$jobdatadir/RG${RG_STEP}/batches" # Make a folder for the batches, combined can stay out later
histdir="$jobdatadir/RG${RG_STEP}/hist" # Make a folder for the histograms
statsdir="$jobdatadir/RG${RG_STEP}/stats"
laundereddir="$jobdatadir/RG${RG_STEP}/laundered"

mkdir -p "$outputdir" "$logsdir" "$jobsdir" # Make these now so that it does it every time we run this job
mkdir -p "$joboutdir" "$jobdatadir" "$batchdir" "$histdir" "$statsdir" "$laundereddir"

PREV_RG=$(( RG_STEP - 1 )) # Store prev iter number for moments later
if [[ $RG_STEP -eq 0 ]]; then
    PREV_Z_HIST=""
else
    PREV_Z_HIST="$jobdatadir/RG${PREV_RG}/hist/z_hist_RG${PREV_RG}_sym.npz"
fi

exec > >(tee -a "$joboutdir/RG_${RG_STEP}.out")
exec 2> >(tee -a "$logsdir/RG_${RG_STEP}.err" >&2)
# Libraries needed
module purge
module load GCC/13.3.0 SciPy-bundle/2024.05
export PYTHONPATH="$codedir:$PYTHONPATH"
cd "$codedir"
SRC_DIR="$codedir/source" # This is where the actual code lives


OUTPUT_T="$histdir/t_hist_RG${RG_STEP}.npz"
OUTPUT_G="$histdir/g_hist_RG${RG_STEP}.npz"
OUTPUT_Z="$histdir/z_hist_RG${RG_STEP}.npz"

echo "Making histograms for RG step $RG_STEP from $NUM_BATCHES batches"

for batch in $(seq 0 $(( NUM_BATCHES - 1 ))); do
    BATCH_DIR="$batchdir/batch_${batch}"

    batch_t=$(ls "$BATCH_DIR"/t_data_RG${RG_STEP}_${BATCH_SIZE}_samples.txt)
    batch_g=$(ls "$BATCH_DIR"/g_data_RG${RG_STEP}_${BATCH_SIZE}_samples.txt)
    batch_z=$(ls "$BATCH_DIR"/z_data_RG${RG_STEP}_${BATCH_SIZE}_samples.txt)

    echo " Adding batch $batch:"
    echo " t dir: $batch_t"
    echo " g dir: $batch_g"
    echo " z dir: $batch_z"

    # 2 choices depending on whether its the first histogram or not
    if [[ ! -f "$OUTPUT_T" || ! -f "$OUTPUT_G" || ! -f "$OUTPUT_Z" ]]; then
        python -m "source.histogram_manager" \
            "$BATCH_SIZE" \
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
            "$BATCH_SIZE" \
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
done

echo "Made histograms at:"
echo "T: $OUTPUT_T"
echo "G: $OUTPUT_G"
echo "Z: $OUTPUT_Z"


symmetrised_z="$histdir/z_hist_RG${RG_STEP}_sym.npz"
python -m "source.helpers" \
    1 \
    "$N" \
    "$OUTPUT_Z" \
    "$symmetrised_z"

# Analyse the moments

stats="$statsdir/z_moments_RG${RG_STEP}_${N}_samples.npz"

if [[ -z "$PREV_Z_HIST" || ! -f "$PREV_Z_HIST" ]]; then
    prev="None"
else
    prev="$PREV_Z_HIST"
fi

python -m "source.rg" \
    "$RG_STEP" \
    "$prev" \
    "$OUTPUT_Z" \
    "$stats"

echo "Moments of histogram for $N samples written to $stats."

for batch in $(seq 0 $(( NUM_BATCHES - 1 ))); do
    launderbatch="$laundereddir/t_laundered_RG${RG_STEP}_batch_${batch}.txt"

    python -m "source.helpers" \
        0 \
        "$BATCH_SIZE" \
        "$symmetrised_z" \
        "$launderbatch"

    echo "Batch $batch of t data laundered from Q(z) saved to $launderbatch"
done


