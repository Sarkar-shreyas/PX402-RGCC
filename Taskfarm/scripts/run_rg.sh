#!/bin/bash

# Set default params
VERSION=2.10
TYPE=FP
METHOD="a"
EXPR="s"
SAMPLER="i"
N=480000000
K=10

# Read command line input
while getopts 'nk:m:e:s:' OPTION; do
    case "${OPTION}" in
        n)
            N=$OPTARG
            echo "Proceeding with $N samples"
            ;;
        k)
            K=$OPTARG
            echo "Proceeding with $K RG steps"
            ;;
        m)
            METHOD="$OPTARG"
            echo "Proceeding with the $METHOD method"
            ;;
        e)
            EXPR=$OPTARG
            echo "Proceeding with ${EXPR}'s expression"
            ;;
        s)
            SAMPLER=$OPTARG
            echo "Proceeding with ${SAMPLER} sampling method"
            ;;
        h)
            echo "==========================================================================="
            echo "                  RG SCRIPT HELPER "
            echo "---------------------------------------------------------------------------"
            echo " -n : Number of samples to generate "
            echo " -k : Number of RG steps to run "
            echo " -m : Method of analysis (a = analytic, n = numerical)"
            echo " -e : Expression to use for analytic method (c = Cain, s = Shaw, j = Jack) "
            echo " -s : Sampler method (r = rejection, i = inverse CDF) "
            echo "==========================================================================="
            echo ""
    esac
done

echo "====================================================="
echo "                  RG WORKFLOW CONFIG "
echo "-----------------------------------------------------"
echo " Version            : $VERSION "
echo " Type               : $TYPE "
echo " Number of samples  : $N "
echo " Number of RG steps : $K "
echo " t' Method          : $METHOD "
echo " Expression         : $EXPR "
echo " Sampler method : $SAMPLER "
echo " Date of job        : [$(date '+%Y-%m-%d %H:%M:%S')] "
echo "====================================================="
echo ""

VERSIONSTR="${VERSION}${EXPR}"

rg_job=$(sbatch --parsable \
        "rg_fp_master.sh" \
        "$VERSIONSTR" "$N" "$K" "$METHOD" "$EXPR" "$SAMPLER")

echo " [$(date '+%Y-%m-%d %H:%M:%S')]: Submitted RG job with id $rg_job "
