#!/bin/bash

shifts=( 0.0 0.005 0.009 )

prev_shift_job=""

for shift in "${shifts[@]}"; do
    if [ -z "$prev_shift_job" ]; then
        shift_job=$(sbatch --parsable \
            shifted_rg.sh \
                "$shift")
        echo "Submitted shift job ${shift_job} for shift $shift"
    else
        shift_job=$(sbatch --parsable \
            --dependency=afterok:${prev_shift_job} \
            shifted_rg.sh \
                "$shift")
        echo "Submitted shift job ${shift_job} for shift $shift (after job ${prev_shift_job})"
    fi

    prev_shift_job="$shift_job"
    sleep 30
done

echo "All shift jobs have been submitted."
