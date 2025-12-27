#!/bin/bash

# Set default params
VERSION=2.10
TYPE=FP
METHOD="a"
EXPR="s"
LAUNDER="i"
SYMMETRISE=1
N=480000000
SEED=10
K=10

set -euo pipefail

# Read command line input
while getopts 'v:n:k:m:e:l:s:y:h' OPTION; do
    case "${OPTION}" in
        v)
            VERSION="$OPTARG" ;;
        n)
            N="$OPTARG" ;;
        k)
            K="$OPTARG" ;;
        m)
            METHOD="$OPTARG" ;;
        e)
            EXPR="$OPTARG" ;;
        l)
            LAUNDER="$OPTARG" ;;
        s)
            SEED="$OPTARG" ;;
        y)
            SYMMETRISE="$OPTARG" ;;
        h)
            echo "============================================================================="
            echo "                              RG SCRIPT HELPER "
            echo "-----------------------------------------------------------------------------"
            echo " -v   : Version/run name "
            echo " -n   : Number of samples to generate "
            echo " -k   : Number of RG steps to run "
            echo " -s   : Starting seed value "
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
            echo "                              RG SCRIPT HELPER "
            echo "-----------------------------------------------------------------------------"
            echo " -v   : Version/run name "
            echo " -n   : Number of samples to generate "
            echo " -k   : Number of RG steps to run "
            echo " -s   : Starting seed value "
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
echo " t' Method          : $METHOD "
echo " Expression         : $EXPR "
echo " Launder method     : $LAUNDER "
echo " Symmetrising?      : $SYMMETRISE "
echo " Date of job        : [$(date '+%Y-%m-%d %H:%M:%S')] "
echo "====================================================="
echo ""

VERSIONSTR="${VERSION}${EXPR}"

rg_job=$(sbatch --parsable \
        "rg_fp_master.sh" \
        "$VERSIONSTR" "$N" "$K" "$SEED" "$METHOD" "$EXPR" "$LAUNDER" "$SYMMETRISE" )

echo " [$(date '+%Y-%m-%d %H:%M:%S')]: Submitted RG job with id $rg_job "
