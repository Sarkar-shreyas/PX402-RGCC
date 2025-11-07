#!/bin/bash
#SBATCH --job-name=send_jobs
#SBATCH --output=../job_outputs/bootstrap/%x_%A_%a.out
#SBATCH --error=../job_logs/bootstrap/%x_%A_%a.err

N=120000000
NUM_RG_ITERS=8

VERSION=1.1
INITIAL=1
EXISTING_T=""
prev_hist_job=""

basedir="$(cd "$SLURM_SUBMIT_DIR/.."&&pwd)" # Root, fyp for now
joboutdir="$basedir/job_outputs/v${VERSION}/output"
datadir="$basedir/job_outputs/v${VERSION}/data"
scriptsdir="$basedir/scripts"
logsdir="$basedir/job_logs/v${VERSION}"
mkdir -p "$datadir" "$logsdir" "$joboutdir"

echo "==================================================="
echo "                  SLURM JOB INFO "
echo "---------------------------------------------------"
echo " Job Name         : $SLURM_JOB_NAME"
echo " Job ID           : $SLURM_JOB_ID"
echo " Submitted from   : $SLURM_SUBMIT_DIR"
echo " Current dir      : $(pwd)"
echo "=================================================="
echo ""

exec > >(tee -a "$joboutdir/JOB${SLURM_JOB_ID}.out")
exec 2> >(tee -a "$logsdir/JOB${SLURM_JOB_ID}.err" >&2)
for step in $(seq 0 $(( NUM_RG_ITERS - 1 ))); do
    echo "[$(date '+%H:%M:%S')]: Proceeding with RG step $step"
    laundereddir="$datadir/RG${step}/laundered"

    if [ "$step" -eq 0 ]; then
        EXISTING_T=""
    else
        prev_step=$(( $step - 1 ))
        EXISTING_T="$datadir/RG${prev_step}/laundered"
    fi

    if [ "$step" -eq 0 ]; then
        gen_job=$(sbatch --parsable \
        --output=../job_outputs/bootstrap/rg_gen_RG${step}_%x_%A_%a.out \
        --error=../job_logs/bootstrap/rg_gen_RG${step}_%x_%A_%a.err \
        "$scriptsdir/rg_gen_batch.sh"\
            "$N" "$INITIAL" "$EXISTING_T" "$step")

        echo "[$(date '+%H:%M:%S')]: Submitted generation job for RG step $step : $gen_job"
    else
        gen_job=$(sbatch --parsable \
        --dependency=afterok:${prev_hist_job} \
        --output=../job_outputs/bootstrap/rg_gen_RG${step}_%A_%a.out \
        --error=../job_logs/bootstrap/rg_gen_RG${step}_%A_%a.err \
        "$scriptsdir/rg_gen_batch.sh"\
            "$N" "$INITIAL" "$EXISTING_T" "$step")

        echo "[$(date '+%H:%M:%S')]: Submitted generation job for RG step $step : $gen_job (after ${prev_hist_job})"
    fi
    echo "---------------------------------------------------"
    hist_job=$(sbatch --parsable \
        --dependency=afterok:${gen_job} \
        --output=../job_outputs/bootstrap/rg_hist_RG${step}_%A_%a.out \
        --error=../job_logs/bootstrap/rg_hist_RG${step}_%A_%a.err \
        "$scriptsdir/rg_hist_manager.sh" \
        "$N" "$step")
    echo "---------------------------------------------------"
    echo "[$(date '+%H:%M:%S')]: Submitted histogram job for RG step $step  : $hist_job (after ${gen_job})"

    prev_hist_job="$hist_job"
    INITIAL=0

    EXISTING_T="$laundereddir"
    echo "==================================================="
done

echo "All ${NUM_RG_ITERS} RG jobs submitted. Final dependency ends at JOB${prev_hist_job}"
