#!/bin/bash

shifts=( 0.0 0.003 0.005 0.007 0.009 )

shift_job=$(sbatch --parsable \
    shifted_rg.sh \
        "${shifts[@]}")
echo "Submitted shift job ${shift_job} for shift ${shifts[0]}, remaining shifts are: ${shifts[*]:1}"

