#!/bin/bash

# Set default params
CONFIG=""
SETS=()
OUT=""
TYPE="QP"
Q_BLOCK_SIZE=10
VARS="p,q"

set -euo pipefail

# Read command line input
while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|--config)
            CONFIG="$2";
            shift 2;;
        --set)
            SETS+=("$2");
            shift 2;;
        -o|--out)
            OUT="$2";
            shift 2;;
        --q-block-size)
            Q_BLOCK_SIZE="$2";
            shift 2;;
        --vars)
            VARS="$2";
            shift 2;;
        -h|--help)
            echo "=============================================================================="
            echo "                              QP TRIAL SCRIPT HELPER "
            echo "------------------------------------------------------------------------------"
            echo " -c | --config   : Config file path "
            echo " --set           : Override settings (Eg; --set 'engine.method = numerical')"
            echo " -o | --out      : Output folder for updated config "
            echo " --q-block-size  : Number of q values per array job (default 10) "
            echo " --vars          : vars to aggregate, comma-separated (default 'p,q') "
            echo " -h | --help     : Help "
            echo "=============================================================================="
            echo "";
            exit 0 ;;
        --)
            shift; break;;
        *)
            echo "Unknown arg $1"
            echo "=============================================================================="
            echo "                              QP TRIAL SCRIPT HELPER "
            echo "------------------------------------------------------------------------------"
            echo " -c | --config   : Config file path "
            echo " --set           : Override settings (Eg; --set 'engine.method = numerical')"
            echo " -o | --out      : Output folder for updated config "
            echo " --q-block-size  : Number of q values per array job (default 10) "
            echo " --vars          : vars to aggregate, comma-separated (default 'p,q') "
            echo " -h | --help     : Help "
            echo "=============================================================================="
            echo "";
            exit 2 ;;
    esac
done

if [[ -z "${CONFIG}" ]]; then
    echo "Missing --config path" >&2
    exit 2
fi


basedir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.."&&pwd)" # Our root directory
codedir="$basedir/code" # Where the code lives
scriptsdir="$basedir/scripts" # Where all shell scripts live

# Libraries needed
module purge
module load GCC/13.3.0 SciPy-bundle/2024.05
source "$basedir/.venv/bin/activate"

export PYTHONPATH="$codedir:$PYTHONPATH"

if (( ${#SETS[@]} )); then
    UPDATED_CONFIG="$(
        python "$codedir/source/parse_config.py" \
        --config "$CONFIG" \
        --type "$TYPE" \
        --set "${SETS[@]}" \
        ${OUT:+--out "$OUT"}
    )"
else
    UPDATED_CONFIG="$(
        python "$codedir/source/parse_config.py" \
        --config "$CONFIG" \
        --type "$TYPE" \
        ${OUT:+--out "$OUT"}
    )"
fi


mkdir -p "$basedir/job_outputs/bootstrap" "$basedir/job_logs/bootstrap"
qp_job=$(sbatch --parsable \
        --output="$basedir/job_outputs/bootstrap/qp_trials_%A.out" \
        --error="$basedir/job_logs/bootstrap/qp_trials_%A.err" \
        "$scriptsdir/qp_trials.sh" \
        "$UPDATED_CONFIG" "$Q_BLOCK_SIZE" "$VARS" )

echo " [$(date '+%Y-%m-%d %H:%M:%S')]: Submitted QP job with id $qp_job "
