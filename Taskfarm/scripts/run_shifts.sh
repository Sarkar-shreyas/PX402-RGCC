#!/bin/bash

# Set default params
VERSION=2.10
TYPE=EXP
METHOD="a"
EXPR="s"
LAUNDER="i"
SYMMETRISE=1
N=480000000
K=10
SEED=10
FP_NUM=$K

shifts=( 0.003 0.004 0.005 0.006 0.0075 0.010 )

set -euo pipefail

# Read command line input
while getopts 'v:n:k:s:f:m:e:l:y:h' OPTION; do
    case "${OPTION}" in
        v)
            VERSION="$OPTARG" ;;
        n)
            N="$OPTARG" ;;
        k)
            K="$OPTARG" ;;
        s)
            SEED="$OPTARG" ;;
        m)
            METHOD="$OPTARG" ;;
        e)
            EXPR="$OPTARG" ;;
        l)
            LAUNDER="$OPTARG" ;;
        y)
            SYMMETRISE="$OPTARG" ;;
        f)
            FP_NUM="$OPTARG" ;;
        h)
            echo "============================================================================="
            echo "                  RG SCRIPT HELPER "
            echo "-----------------------------------------------------------------------------"
            echo " -v   : Version/run name "
            echo " -n   : Number of samples to generate "
            echo " -k   : Number of RG steps to run "
            echo " -s   : Starting seed value "
            echo " -f   : Step no. for FP distribution "
            echo " -m   : Method of computation (a = analytic, n = numerical)"
            echo " -e   : Expression to use for analytic method (c = Cain, s = Shaw, j = Jack) "
            echo " -l   : Launder method (r = rejection, i = inverse CDF) "
            echo " -y   : Symmetrise z data per step? (1 = yes, 0 = no)"
            echo " -h   : help "
            echo "============================================================================="
            echo "";
            exit 0 ;;
        *)
            echo "============================================================================="
            echo "                  RG SCRIPT HELPER "
            echo "-----------------------------------------------------------------------------"
            echo " -v   : Version/run name "
            echo " -n   : Number of samples to generate "
            echo " -k   : Number of RG steps to run "
            echo " -s   : Starting seed value "
            echo " -f   : Step no. for FP distribution "
            echo " -m   : Method of computation (a = analytic, n = numerical)"
            echo " -e   : Expression to use for analytic method (c = Cain, s = Shaw, j = Jack) "
            echo " -l   : Launder method (r = rejection, i = inverse CDF) "
            echo " -y   : Symmetrise z data per step? (1 = yes, 0 = no)"
            echo " -h   : help "
            echo "============================================================================="
            echo "";
            exit 2 ;;
    esac
done

echo "====================================================="
echo "                  RG WORKFLOW CONFIG "
echo "-----------------------------------------------------"
echo " Version            : $VERSION "
echo " Type               : $TYPE "
echo " Number of samples  : $N "
echo " Number of RG steps : $K "
echo " Starting seed      : $SEED "
echo " Prev FP step no.   : $FP_NUM "
echo " t' Method          : $METHOD "
echo " Expression         : $EXPR "
echo " Launder method     : $LAUNDER "
echo " Symmetrising?      : $SYMMETRISE "
echo " Shifts to apply    : ${shifts[*]} "
echo " Date of job        : [$(date '+%Y-%m-%d %H:%M:%S')] "
echo "====================================================="
echo ""

VERSIONSTR="${VERSION}${EXPR}"

shift_job=$(sbatch --parsable \
    shifted_rg.sh \
        "$VERSIONSTR" "$N" "$K" "$SEED" "$FP_NUM" "$METHOD" "$EXPR" "$LAUNDER" "$SYMMETRISE" "${shifts[@]}")
echo "Submitted shift job ${shift_job} for shift ${shifts[0]}, remaining shifts are: ${shifts[*]:1}"
