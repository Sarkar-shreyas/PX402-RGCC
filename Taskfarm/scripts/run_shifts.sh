#!/bin/bash
##
## PURPOSE: Submit an Exponential-perturbation (EXP) shifted-RG job to the Slurm scheduler
##          on vulcan2. Applies a small shift to the FP distribution and runs further RG
##          steps to extract the critical exponent ν. Validates/updates the YAML config via
##          parse_config.py then hands off to shifted_rg.sh, which owns the Slurm resource
##          directives.
##
## USAGE:
##   bash run_shifts.sh -c <config.yaml> [-i <shift_index>] [--set "key.path=value" ...] [-o <out_dir>]
##
##   Example (from CLAUDE.md):
##     bash Taskfarm/scripts/run_shifts.sh \
##       --config Taskfarm/configs/iqhe.yaml \
##       --index 0 \
##       --out /tmp/configs
##
## SLURM RESOURCES: Resource directives (--nodes, --ntasks, --mem, --time, etc.) are
##   declared inside shifted_rg.sh, not here. This script is only an orchestration
##   wrapper that resolves the config path and passes the shift index before calling sbatch.
##
## MODULES LOADED:
##   GCC/13.3.0           — C/Fortran toolchain required by NumPy/SciPy native extensions.
##   SciPy-bundle/2024.05 — Provides NumPy, SciPy, and related stack at the correct ABI.
##   .venv                — Project virtualenv activated on top of the module environment
##                          to pick up PyYAML and any other pip-only dependencies.
##
## INPUT:
##   -c / --config  : Path to the YAML config file (required).
##   -i / --index   : Integer index into the config's `data_settings.shifts` array
##                    (default 0). Selects which perturbation magnitude to use.
##   --set          : Zero or more "key.path=value" overrides applied before submission.
##   -o / --out     : Optional directory to write the resolved config; defaults to a
##                    temp path chosen by parse_config.py.
##
## OUTPUT:
##   job_outputs/bootstrap/shifted_rg_<jobid>.out  — Slurm stdout for the shifted-RG job.
##   job_logs/bootstrap/shifted_rg_<jobid>.err     — Slurm stderr for the shifted-RG job.
##   Both directories are created by this script with mkdir -p before sbatch is called.
##

# Set default params
CONFIG=""
SETS=()
OUT=""
SHIFT_INDEX=0
TYPE="EXP"
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
        -i|--index)
            SHIFT_INDEX="$2";
            shift 2;;
        -h|--help)
            echo "=============================================================================="
            echo "                              RG SCRIPT HELPER "
            echo "------------------------------------------------------------------------------"
            echo " -c | --config   : Config file path "
            echo " --set           : Override settings ( Eg; --set 'engine.method = numerical' )"
            echo " -o | --out      : Output folder for updated config "
            echo " -i | --index    : Index of shifts array to use"
            echo " -h | --help     : Help "
            echo "=============================================================================="
            echo "";
            exit 0 ;;
        --)
            shift; break;;
        *)
            echo "Unknown arg $1"
            echo "=============================================================================="
            echo "                              RG SCRIPT HELPER "
            echo "------------------------------------------------------------------------------"
            echo " -c | --config   : Config file path "
            echo " --set           : Override settings ( Eg; --set 'engine.method = numerical' )"
            echo " -o | --out      : Output folder for updated config "
            echo " -i | --index    : Index of shifts array to use"
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
module purge                                   # Clear any inherited module state for a clean environment
module load GCC/13.3.0 SciPy-bundle/2024.05   # Load the HPC-provided NumPy/SciPy stack
source "$basedir/.venv/bin/activate"           # Activate the project venv on top (for PyYAML etc.)

# Expose the project source tree to Python so imports resolve without installation
export PYTHONPATH="$codedir:$PYTHONPATH"

# Validate the YAML config (and apply any --set overrides) before handing the
# resolved path to sbatch. parse_config.py prints the final config path to stdout.
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

shift_job=$(sbatch --parsable \            # --parsable: print only the numeric job ID so it can be captured in $shift_job
        --output="$basedir/job_outputs/bootstrap/shifted_rg_%A.out" \   # %A expands to the master job ID
        --error="$basedir/job_logs/bootstrap/shifted_rg_%A.err" \
        "$scriptsdir/shifted_rg.sh" \
        "$UPDATED_CONFIG" "$SHIFT_INDEX")  # SHIFT_INDEX selects the perturbation from data_settings.shifts
echo " [$(date '+%Y-%m-%d %H:%M:%S')]: Submitted RG Exp job with id $shift_job "
