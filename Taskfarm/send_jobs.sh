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

exec > >(tee -a "$joboutdir/RG_${RG_STEP}.out")
exec 2> >(tee -a "$logsdir/RG_${RG_STEP}.err" >&2)
for step in $(seq 0 $(( NUM_RG_ITERS - 1 ))); do
    echo "Submitted data gen for RG step $step"
    laundereddir="$datadir/RG${step}/laundered"

    if [ "$step" -eq 0 ]; then
        EXISTING_T=""
    else
        prev_step=$(( $step - 1 ))
        EXISTING_T="$datadir/RG${prev_step}/laundered"
    fi

    if [ "$step" -eq 0 ]; then
        gen_job=$(sbatch --parsable "$scriptsdir/rg_gen_batch.sh"\
            "$N" "$INITIAL" "$EXISTING_T" "$step")
    else
        gen_job=$(sbatch --parsable --dependency=afterok:${prev_hist_job} \
        "$scriptsdir/rg_gen_batch.sh"\
            "$N" "$INITIAL" "$EXISTING_T" "$step")
    fi
    echo "Submitted generation job: $gen_job"

    hist_job=$(sbatch --parsable --dependency=afterok:${gen_job} "$scriptsdir/rg_hist_manager.sh" \
        "$N" "$step")
    echo "Submitted histogram job: $hist_job (after $gen_job)"

    prev_hist_job="$hist_job"
    INITIAL=0

    EXISTING_T="$laundereddir"
done
