#!/usr/bin/env python
"""Build and append compressed NPZ histograms for the t and z RG variables.

Purpose
-------
Converts raw ``.npy`` sample arrays (amplitude ``t`` or log-ratio ``z``)
into compressed ``.npz`` histogram archives, or accumulates new samples
into an existing archive.  Used by both FP and EXP Slurm batch scripts as
well as the local IQHE driver.

PROCESS modes
-------------
- **PROCESS 0 — Initialise**: creates a brand-new histogram from a single
  ``.npy`` input file.  Bin count and range are read from the ``RG_CONFIG``
  singleton (``z_bins`` / ``z_range`` for z, ``t_bins`` / ``t_range`` for
  t).  For EXP runs the optional ``SHIFT`` argument offsets the z-range so
  that the histogram is centred on the perturbed distribution rather than
  the fixed-point distribution.
- **PROCESS 1 — Append**: loads an existing ``.npz`` archive, histograms
  the new input data using the *same bin edges* already stored in the
  archive, and writes the summed counts back to the output file.  The
  existing file is not modified in place; counts are summed and saved to
  ``OUTPUT_FILE`` (which may be the same path).

NPZ schema
----------
Every ``.npz`` file written by this module contains exactly three arrays:

``histval``
    Integer or float64 count array of shape ``(n_bins,)``.
``binedges``
    Float64 bin-edge array of shape ``(n_bins + 1,)``.
``bincenters``
    Float64 bin-centre array of shape ``(n_bins,)``; midpoints of
    consecutive edges.

Load with ``np.load(path, allow_pickle=False)``.

CLI usage
---------
Initialise (PROCESS 0)::

    python -m source.histogram_manager 0 VAR INPUT_FILE OUTPUT_FILE RG_STEP [SHIFT]

Append (PROCESS 1)::

    python -m source.histogram_manager 1 VAR INPUT_FILE EXISTING_FILE OUTPUT_FILE RG_STEP [SHIFT]

Arguments
---------
PROCESS : int
    ``0`` to create a new histogram; ``1`` to append to an existing one.
VAR : str
    Variable name — ``"t"`` or ``"z"``.
INPUT_FILE : str
    Path to the ``.npy`` sample array to histogram.
EXISTING_FILE : str
    (PROCESS 1 only) Path to the existing ``.npz`` archive whose bin edges
    are reused and whose counts are accumulated into.
OUTPUT_FILE : str
    Destination ``.npz`` path for the result (created or overwritten).
RG_STEP : int
    RG iteration index — used for log messages only.
SHIFT : str, optional
    Numeric shift (as a string) applied to the z-range boundaries.
    Marks EXP-run histograms that are centred on a perturbed distribution.
"""

import numpy as np

from source.utilities import (
    save_data,
)
from source.config import get_rg_config
import sys
from datetime import datetime, timezone


def _bin_and_range_manager(
    var: str, hist_vars: dict, shift: str | None = None
) -> tuple:
    """Select bin count and range for variable 't' or 'z'.

    Parameters
    ----------
    var : str
        Variable name, expected to be 't' or 'z' (case insensitive).
    shift : str or None
        Optional numeric shift (as string) to add to the z-range. If
        provided, the returned z-range will be shifted by this amount.

    Returns
    -------
    tuple
        (bins, range) where `bins` is an integer and `range` is a tuple
        (min, max) suitable for `numpy.histogram`.

    Raises
    ------
    ValueError
        If `var` is not one of 't' or 'z'.
    """
    if var != "t" and var != "z":
        raise ValueError(f"Invalid variable entered: {var}. Expected 't' or 'z'")
    bins = hist_vars[var]["bins"]
    range = hist_vars[var]["range"]
    if var == "z" and shift is not None:
        shift_val = float(shift.strip())
        min_z, max_z = range
        min_z += shift_val
        max_z += shift_val
        range = (min_z, max_z)
    return bins, range


def construct_initial_histogram(
    data: np.ndarray,
    output_filename: str,
    var: str,
    hist_vars: dict,
    shift: str | None = None,
) -> None:
    """Create a new NPZ histogram archive from a raw sample array.

    Bin count and range are resolved via ``_bin_and_range_manager`` using
    ``hist_vars``, which is populated from the ``RGConfig`` singleton
    (``z_bins`` / ``z_range`` for z, ``t_bins`` / ``t_range`` for t).
    For EXP runs, passing ``shift`` offsets the z-range boundaries by that
    amount so the histogram spans the perturbed distribution rather than
    the fixed-point distribution — this is what distinguishes EXP histogram
    files from FP ones at the same RG step.

    Args:
        data: 1-D sample array to histogram.  Non-finite values are dropped
            before binning.
        output_filename: Destination ``.npz`` path.  Written (or
            overwritten) by ``save_data``; numpy appends ``.npz`` if
            absent.
        var: Variable name — ``"t"`` or ``"z"`` (case-insensitive).
            Selects the binning parameters from ``hist_vars``.
        hist_vars: Mapping of variable name to ``{"bins": int, "range":
            tuple}`` entries.  Typically built from ``rg_config.z_bins``,
            ``rg_config.z_range``, ``rg_config.t_bins``, and
            ``rg_config.t_range``.
        shift: Numeric shift as a string (e.g. ``"0.005"``).  When
            provided and ``var == "z"``, the z-range min and max are each
            incremented by this value before binning.  ``None`` (default)
            leaves the range unchanged.
    """
    # Drop nan values
    data = data[np.isfinite(data)]
    # Get bins and range for this variable and shift
    var = var.strip().lower()
    bins, range = _bin_and_range_manager(var, hist_vars, shift)

    # Bin the samples into a count histogram; bin_edges has length bins+1.
    hist_vals, bin_edges = np.histogram(data, bins=bins, range=range)
    # Bin centres are the midpoints of consecutive edges.
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    save_data(
        hist_vals,
        bin_edges,
        bin_centers,
        output_filename,
    )


def append_to_histogram(
    input_data: np.ndarray,
    existing_file: str,
    output_file: str,
    range: tuple,
) -> None:
    """Accumulate new sample counts into an existing NPZ histogram archive.

    The existing archive is loaded read-only; its ``binedges`` array is
    reused directly as the bin specification for the new data so that both
    arrays share identical boundaries.  The new per-bin counts are then
    added element-wise to the existing ``histval`` array and the result is
    written to ``output_file`` — which may be the same path as
    ``existing_file``, effectively updating it in place.

    Bin edges are not recomputed here — they are inherited from the archive
    created by ``construct_initial_histogram``.  ``range`` is passed in
    pre-computed (via ``_bin_and_range_manager``) and applied to
    ``np.histogram`` so that out-of-range samples are excluded consistently
    with the original build step.  For EXP runs the shift has already been
    baked into ``range`` by the caller, preserving EXP histogram provenance
    without any special handling inside this function.

    Args:
        input_data: 1-D sample array to accumulate.  Non-finite values are
            dropped before binning.
        existing_file: Path to the ``.npz`` archive whose ``histval``,
            ``binedges``, and ``bincenters`` arrays are loaded and
            accumulated into.  The file is not modified; the result is
            written to ``output_file``.
        output_file: Destination ``.npz`` path for the updated histogram
            (created or overwritten).
        range: ``(min, max)`` tuple passed to ``np.histogram`` as the
            ``range`` argument.  Must be consistent with the edges already
            stored in ``existing_file``.

    Raises:
        ValueError: If the number of bins in the newly computed histogram
            does not match the number of bins in the existing archive,
            indicating incompatible bin-edge arrays.
    """
    # Drop nan values
    input_data = input_data[np.isfinite(input_data)]

    # Load the target file, should be an .npz file
    existing_data = np.load(existing_file, allow_pickle=False)
    existing_vals = existing_data["histval"]
    existing_bin_edges = existing_data["binedges"]
    existing_bin_centers = existing_data["bincenters"]

    # Re-use the stored bin edges so new counts land in identical bins;
    # density=False keeps raw integer counts for accumulation.
    data_counts, _ = np.histogram(
        input_data, bins=existing_bin_edges, range=range, density=False
    )
    # Guard: bin counts must match before accumulation or the addition is meaningless.
    if data_counts.size != existing_vals.size:
        raise ValueError(
            f"Histogram sizes mismatched: Input: {data_counts.size}, Existing: {existing_vals.size}"
        )

    # Element-wise addition accumulates the new counts into the running total.
    existing_vals += data_counts
    save_data(existing_vals, existing_bin_edges, existing_bin_centers, output_file)


if __name__ == "__main__":
    input_length = len(sys.argv)
    if input_length not in [6, 7, 8]:
        raise SystemExit(
            " Usage: histogram_manager.py PROCESS VAR_NAME INPUT_FILE [EXISTING_FILE] OUTPUT_FILE RG_STEP [SHIFT] \n"
            " PROCESS 0 : Initialise histogram for input variable \n"
            " PROCESS 1 : Append input data to existing histogram "
        )
    process = int(sys.argv[1].strip())
    var_name = sys.argv[2].strip().lower()
    input_file = sys.argv[3].strip()
    rg_config = get_rg_config()
    hist_vars = {
        "z": {"bins": rg_config.z_bins, "range": rg_config.z_range},
        "t": {"bins": rg_config.t_bins, "range": rg_config.t_range},
    }
    if process == 0:
        # Then we're making the initial histogram, so there's no existing file input
        output_file = sys.argv[4].strip()
        rg_step = int(sys.argv[5].strip())
        mode = "Initialise"
        if input_length == 7:
            shift = sys.argv[6].strip()
        else:
            shift = None
    elif process == 1:
        # Then we're appending to an existing histogram
        existing_file = sys.argv[4].strip()
        output_file = sys.argv[5].strip()
        rg_step = int(sys.argv[6].strip())
        mode = "Append"
        if input_length == 8:
            shift = sys.argv[7].strip()
        else:
            shift = None
    else:
        raise SystemExit(
            "Invalid process entered. Process must be either 0 (Build new hist) or 1 (Append to existing hist)"
        )
    print("-" * 100)
    if shift is not None and len(shift) == 0:
        shift = None
    current_date = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    print(
        f"[{current_date}] : Histogram job of [RG step {rg_step}] with mode [{mode}] started for var {var_name} and shift {shift}"
    )
    data = np.load(input_file)
    if process == 0:
        construct_initial_histogram(data, output_file, var_name, hist_vars, shift)
        print(f"Histogram saved to {output_file}")
    else:
        if not existing_file:
            raise SystemExit(f"No existing histogram was found for mode {mode}")
        else:
            bins, range = _bin_and_range_manager(var_name, hist_vars, shift)
            append_to_histogram(data, existing_file, output_file, range)
            print(f"Appended input data to existing data at {existing_file}")

    print("-" * 100)
