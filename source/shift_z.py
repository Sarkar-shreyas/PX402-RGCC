"""Create perturbed t-samples for EXP (shifted) RG runs.

Physical motivation
-------------------
At the RG fixed point the z-distribution Q*(z) is symmetric about z = 0.
To extract the critical exponent ν, EXP runs measure how quickly a small
perturbation to this fixed-point distribution grows under repeated RG
iterations.  By adding a constant ``SHIFT`` to every z-sample drawn from
Q*(z), this script constructs an initial condition that is displaced from
the fixed point by a known amount.  The growth rate of that displacement
over RG steps is related to ν via

    λ = 1 / ν

where λ is the leading relevant RG eigenvalue measured from the EXP runs.

Relationship to the EXP run type
---------------------------------
Each EXP run is seeded with the output of this script rather than with a
laundered FP distribution.  A separate EXP run is executed for each shift
value in the ``shifts`` list (config key ``data_settings.shifts``;
typical values ``0.003``, ``0.005``, ``0.007``, ``0.009``).  The shift
magnitude is in the same units as z (dimensionless log-ratio) and must be
small enough to stay in the linear regime near the fixed point while
remaining large enough to be measurable over a finite number of RG steps.

How the output feeds into the EXP pipeline
-------------------------------------------
The output ``.npy`` file is a 1-D float64 array of ``NUM_SAMPLES``
amplitude values ``t ∈ (0, 1)``.  It is passed as ``EXISTING_T_FILE`` to
the first RG step of the corresponding EXP Slurm job array, exactly as a
continuation run would use the output of a prior FP step.  Subsequent EXP
RG steps chain from the output of the preceding step in the normal way.

The bin edges of the input histogram are **not** modified; only the
resampled z-values are translated.  The EXP histogram files built from
these samples use a shifted ``z_range`` (via the ``SHIFT`` argument to
``histogram_manager.py``) so that the displaced distribution is captured
within the histogram bounds.

CLI usage
---------
::

    python -m source.shift_z NUM_SAMPLES INPUT_HIST SEED SHIFT OUTPUT_T

Arguments
---------
NUM_SAMPLES : int
    Number of amplitude samples to generate.
INPUT_HIST : str
    Path to the symmetrised FP z-histogram ``.npz`` archive (must contain
    ``histval``, ``binedges``, and ``bincenters``).
SEED : int
    Integer seed for the NumPy PCG64 RNG; controls the laundering draw.
SHIFT : float
    Constant perturbation added to every z-sample, in the same units as z
    (dimensionless log-ratio).  Typical EXP values: ``0.003``–``0.009``.
    Positive shifts move the distribution towards the insulating phase
    (larger z); negative towards the metallic phase.
OUTPUT_T : str
    Destination ``.npy`` path for the resulting float64 amplitude array.
"""

import numpy as np
from source.utilities import launder, convert_z_to_t, build_rng
from source.config import get_rg_config
import sys

if __name__ == "__main__":
    num_samples = int(sys.argv[1].strip())
    input_file = sys.argv[2].strip()
    output_file = sys.argv[3].strip()
    seed = int(sys.argv[4].strip())
    shift = float(sys.argv[5].strip())
    rng = build_rng(seed)
    perturbation = shift
    rg_config = get_rg_config()
    sampler = rg_config.resample
    sym_z = np.load(input_file)
    sym_hist_vals = sym_z["histval"]
    sym_bins = sym_z["binedges"]
    sym_centers = sym_z["bincenters"]

    print("-" * 100)
    print(f"Loaded z histogram from {input_file}")

    # Resample num_samples z-values from Q*(z) via inverse-CDF laundering;
    # the returned array is 1-D float64 with values in the histogram's z-range.
    sym_sample = launder(
        num_samples, sym_hist_vals, sym_bins, sym_centers, rng, sampler
    )

    print(f"Laundered {num_samples} samples from loaded z histogram")

    # Translate every z-sample by the perturbation magnitude.  This rigid
    # shift of the distribution is the EXP initial condition; the bin edges
    # of the source histogram are unchanged — only the sampled values move.
    shifted_sample = sym_sample + perturbation
    # Map shifted z back to amplitude t via t = sqrt(1 / (1 + exp(z))).
    shifted_t = convert_z_to_t(shifted_sample)
    print(f"Shifted laundered sample by {perturbation}")
    np.save(output_file, shifted_t)

    print(f"Shifted t sample has been saved to {output_file}")

    print("-" * 100)
