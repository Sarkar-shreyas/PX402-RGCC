#!/usr/bin/env python
# coding: utf-8

# # QSHE Data Analysis Notebook
# 
# This notebook is the sole analysis environment for Quantum Spin Hall Effect (QSHE) data produced by
# the taskfarm pipeline. It loads aggregated q-p trial data from HPC runs, post-processes RG flow
# trajectories across parameter space, visualises velocity fields, phase boundaries, and density
# distributions, and extracts the critical exponent ν characterising the QSHE phase transition via
# the effective-nu formula νeff = 1 / ((Δslope / ln 2) + 1).
# 
# | Section | Description |
# |---------|-------------|
# | **Initialisation** | Imports, RNG seeding (seed=1234), project-module setup |
# | **Helper functions** | `load_data`, `plot_mets`, `get_json_data` — data loading and metric plotting utilities |
# | **Post-Taskfarm implementation** | Config loading and data-structure initialisation for taskfarm outputs |
# | **Taskfarm data analysis** | Loads aggregated `p_data_agg.npy` / `q_data_agg.npy`; computes RG velocity fields |
# | **Field analysis — Streamplots** | Vector-field streamplots of RG flow over (p, q) parameter space |
# | **Check grid flow** | Gridded RG flow trajectories from initialisation through final RG steps |
# | **Velocity heatmap** | Heatmaps of \|dp/dn\| and \|dq/dn\| showing flow speed per RG iteration |
# | **Check candidate FPs and eigenvalues** | Fixed-point candidates and Jacobian eigenvalue stability analysis |
# | **Boundary plots** | Phase-boundary maps across RG steps |
# | **Density plots** | Parameter-value distributions at each RG step |
# | **Gamma Analysis** | Gamma-crossing analysis across RG steps |
# | **Nu / Critical exponent** | Computes ν vs q, ν vs system size, and produces final critical-exponent plots |
# | **Disorder potential** | Chalker–Coddington disorder potential and S-matrix visualisations |
# | **Landau and Conductance** | Landau band structure and conductance plateau plots |
# | **Deprecated** | Archived analysis code, no longer in active use |
# 
# **Dependencies:** </br>`numpy`, `matplotlib`, `scipy` (interpolate, stats), `json`, `os`, `collections`;</br>
# project modules: `constants`, `source.utilities`, `source.config`, `QSHE.testing_qshe`, `Local.run_local_qshe`
# 
# **Inputs:** </br>`{DATA_DIR}/{dataversion}/QP/data/p_data_agg.npy`,</br>
# `{DATA_DIR}/{dataversion}/QP/data/q_data_agg.npy`,</br>
# `{DATA_DIR}/{dataversion}/QP/config/updated_config.yaml`
# (current version: `qp_unfixed_numerical_shreyas`)
# 
# **Outputs:** </br>RG flow diagrams, velocity heatmaps, boundary and density plots, gamma-crossing plots,</br>
# ν vs q (`nu_vs_q.png`, `qshe_nu_q0.pdf`), ν vs system-size plots, </br>Landau band and conductance
# plateau figures; all saved under `{DATA_DIR}/{dataversion}/QP/plots/` and `./report/`
# 

# ## Initialisation

# In[1]:


# Simple module imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import json
import os
# from numpy.polynomial import polynomial
# from scipy.stats import norm
from scipy.interpolate import make_splprep
from scipy.stats import iqr
# Lazy imports - comment out when not in use
from constants import *
from source.utilities import *
from source.config import *
# from analysis.data_plotting import *
# from analysis.critical_exponent import *
from QSHE.testing_qshe import *
from Local.run_local_qshe import *


rng = build_rng(1234)


# ### Helper functions for the notebook

# In[2]:


# Setup vars
from collections import defaultdict
theta_vals = range(1, 7)
thetas = {"1": 0, "2": r"$\pi/8$", "3": r"$3/pi/16$", "4": r"$/pi/4$", "5": r"$3/pi/8$", "6": r"$\pi/2$"}

def find_intersections(xvals, ydict, totalsteps, metrictype = "mean"):
    """
    Given an array of x values and multiple y-value arrays (over the same x-axis), obtain the (x,y) coordinate where all lines intersect.
    Due to discretisation error, we set a tolerance for 'intersection' in y
    We compute the mean y value per x value, find the variance for all steps, and see whether there exists points where the variance of all curves falls below our tolerance
    """
    # We expect the ydict to be of the form ydict[xval][step] = (mean, median)
    # And the metrictype will tell us which value to use for checking
    variances = {}
    for xval in xvals:
        # First we obtain all the yvalues for a particular xval
        yvals = []
        for step in range(totalsteps):
            ymean, ymedian = ydict[xval][step][0]
            if metrictype == "mean":
                yvals.append(ymean)
            elif metrictype == "median":
                yvals.append(ymedian)
            else:
                raise ValueError(f"{metrictype} is an invalid or inaccurate metric to use. Use 'mean' or 'median'.")
        yarray = np.array(yvals)
        # Then we find its mean and variance
        yarray_var = np.var(yarray)
        variances.update({xval:yarray_var})
    return variances


def close_old_plots():
    """Close all old open figures"""
    openfigs = plt.get_fignums()
    for fignum in openfigs:
        plt.close(fignum)

def convert_z_to_x(z):
    return np.arcsinh(np.exp(z/2))

def convert_g_to_x(g, theta=0.0):
    z = convert_g_to_z(g)
    return convert_z_to_x(z)

def convert_x_to_z(x):
    return np.log(np.sinh(x)**2)

def convert_x_to_g(x):
    z = convert_x_to_z(x)
    return convert_z_to_g(z)

def load_var_moments(rg_steps, var, val):
    # Load prev files with varying thetas
    filename = f"{local_dir}/archived data/theta test/theta_test_{val}_numerical_shreyas/FP/stats/{var}_moments.json"
    moments = []
    with open(filename, "r") as file:
        data = json.load(file)
    for step in range(rg_steps):
        try:
            moments.append(data[f"RG_{step}"]["mean"])
        except KeyError:
            continue
    return moments

def get_median_and_mode(rg_steps, var, version, thetanum):
    """Compute the median and mode of histogram data"""
    foldername = f"{local_dir}/theta_{thetanum}/{version}/FP/hist/{var}"
    medians = []
    modes = []
    for step in range(rg_steps):
        try:
            filename = f"{foldername}/{var}_hist_RG{step}.npz"
            data = np.load(filename)
        except FileNotFoundError:
            continue
        medians.append(np.median(data["histval"]))
        mode_index = np.argmax(data["histval"])
        mode = data["bincenters"][mode_index]
        modes.append(mode)
    return medians, modes


def load_fixed_theta_moments(rg_steps, fixedvar, thetanum, fixedvarval, var):
    """Load moments from old trials at a fixed theta"""
    filename = f"{local_dir}/theta_{thetanum}/theta_{thetanum}_{fixedvar}{fixedvarval}_numerical_shreyas/FP/stats/{var}_moments.json"
    moments = []
    with open(filename, "r") as file:
        data = json.load(file)
    for step in range(rg_steps):
        try:
            moments.append(data[f"RG_{step}"]["mean"])
        except KeyError:
            continue
    return moments

def load_hists(rg_steps, fixedvar, thetanum, fixedvarval, var):
    """Load histograms from theta trials"""
    filename = f"{local_dir}/theta_{thetanum}_{fixedvar}{fixedvarval}_numerical_shreyas/FP/hist/{var}/{var}_hist"
    histdata = []
    for step in range(rg_steps):
        try:
            histfile = f"{filename}_RG{step}.npz"
            ht = np.load(histfile)
            histdata.append(ht)
        except KeyError or FileNotFoundError:
            continue
    return histdata

def collapse_data(data_dict):
    """Collapse nested dictionaries into a json saveable format"""
    if isinstance(data_dict, dict):
        return {collapse_data(key): collapse_data(value) for key, value in data_dict.items()}
    elif isinstance(data_dict, np.ndarray):
        return data_dict.tolist()
    else:
        return data_dict

def save_metric_json(data, filename):
    """Save input data into a json file"""
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

def build_state_dict(qvals, gvals, nsamples, steps, metric, fix=True):
    """Build a state dict containing relevant config params"""
    state = {}
    num_qs = int(len(qvals))
    num_gs = int(len(gvals))
    min_q = float(min(qvals))
    max_q = float(max(qvals))
    min_g = float(min(gvals))
    max_g = float(max(gvals))
    trimmed_qs = [round(float(q), 3) for q in qvals]
    trimmed_gs = [round(float(g), 3) for g in gvals]
    state.update({"q":{"Num": num_qs, "Max": round(max_q, 3), "Min": round(min_q, 3), "Data": trimmed_qs}})
    state.update({"g":{"Num": num_gs, "Max": round(max_g, 3), "Min": round(min_g, 3), "Data": trimmed_gs}})
    state.update({"data": {"type": metric, "samples": nsamples, "steps": steps, "fixed": fix}})
    return state

def quicker_trials(qs, phis, g_init, samples, nsteps, metric = "mean", var = "g", fix = True):
    gmeans = defaultdict(list)
    qmeds = defaultdict(list)
    for q in qs:
        t_init = rng.uniform(np.sqrt(g_init-1e-5), np.sqrt(g_init+1e-5), samples)
        f_init = q*(1 - t_init**2)
        # print(np.mean(t_init), np.mean(f_init), g_init)
        for step in range(nsteps):
            indices=  rng.integers(0, samples, (samples, 5))
            ts = np.take(t_init, indices)
            fs = np.take(f_init, indices)
            tp = numerical_solver(ts, fs, phis, samples, 2, [1.0,0.0,0.0,0.0], samples)
            tp = np.clip(tp, 1e-6, 1.0-1e-6)
            p = tp**2
            if not fix:
                fp = numerical_solver(ts, fs, phis, samples, 17, [1.0,0.0,0.0,0.0], samples)
            else:
                fp = np.sqrt((1-p)*q)
            # fp = np.clip(fp, 1e-6, 1.0-1e-6)
            f2 = fp**2
            g = p + f2
            tmax = np.max(tp)
            tmin = np.min(tp)
            fmax = np.max(fp)
            fmin = np.min(fp)
            if tmax > 1.0 or fmax > 1.0 or np.max(f2) > 1.0:
                print(f"tmax = {tmax}, fmax = {fmax}. At step {step} for ginit = {g_init} and q = {q}")
            if tmin < 0.0 or fmin < 0.0 or np.min(f2) < 0.0:
                print(f"tmax = {tmax}, fmax = {fmax}. At step {step} for ginit = {g_init} and q = {q}")
            # g = np.clip(g, 1e-12, 1.0-1e-12)
            # tp= np.clip(tp, 1e-12, 1-1e-12)
            # fp = np.clip(fp, 1e-12, 1-1e12)
            if var == "z":
                gp = convert_g_to_z(g)
                # gp = np.log((1 - g - f2)/g)
            elif var == "x":
                gp = convert_g_to_x(g)
            elif var == "p":
                gp = p
            else:
                gp = g
            unique, counts = np.unique(gp, return_counts=True)
            maxunique = np.argmax(counts)
            mode = unique[maxunique]
            median = np.median(gp)
            mean = np.mean(gp)
            qmed = np.median(f2/(1-p))
            qmean = np.mean(f2/(1-p))
            qmeds[step].append((qmean, qmed))
            if metric == "mode":
                gmeans[step].append(mode)
            elif metric == "median":
                gmeans[step].append(median)
            elif metric == "all":
                gmeans[step].append((mean, median))
            else:
                gmeans[step].append(mean)
            t_init = tp
            f_init = fp
    # print(q, qmeds)
    return gmeans, qmeds

# Lets try to find the point where the shift between subsequent RG steps change
# Pick two consecutive RG steps to analyse
def get_meds(step_a, step_b, data_dict, qarray, garray, var = "p", fromjson = False):
    medgs_a = {}
    medgs_b = {}
    meangs_a = {}
    meangs_b = {}
    for q in qarray[:]:
        # Other data extraction method if we load the json data instead
        if fromjson:
            gs = data_dict[f"{q}"][var]
            step_a_meds = gs[f"{step_a}"]["Median"]
            step_b_meds = gs[f"{step_b}"]["Median"]
            step_a_means = gs[f"{step_a}"]["Mean"]
            step_b_means = gs[f"{step_b}"]["Mean"]
        else:
            # Otherwise, we want to generate arrays of the mean/median for all ginits, for two consecutive RG steps.
            gs = data_dict[q]
            step_a_meds = []
            step_b_meds = []
            step_a_means = []
            step_b_means = []
            for ginit in garray:
                meang_a, medg_a = gs[ginit][step_a][0]
                meang_b, medg_b = gs[ginit][step_b][0]
                step_a_meds.append(medg_a)
                step_b_meds.append(medg_b)
                step_a_means.append(meang_a)
                step_b_means.append(meang_b)

        medgs_a.update({q:np.array(step_a_meds)})
        medgs_b.update({q:np.array(step_b_meds)})
        meangs_a.update({q:np.array(step_a_means)})
        meangs_b.update({q:np.array(step_b_means)})
    return medgs_a, medgs_b, meangs_a, meangs_b

def plot_delta_med(gmeds_a, gmeds_b, qarray, garray, step1, step2, var, metrictype, plotfolder, qstep, n, save = True):
    close_old_plots()
    ylab= f"$\\Delta {{{var}}}_{{{metrictype}}}$"
    xlab = f"${{{var}}}_{{init}}$"
    plt.figure(figsize=(12, 8))
    plt.axhline(y=0.0, linestyle="--", color="r", alpha=0.4)
    deltas = {q:[] for q in qarray}
    for q in qarray[::qstep]:
        crossover_index = 0
        deltag_mean = gmeds_b[q] - gmeds_a[q]
        # deltag_mean = meangs_b[q] - meangs_a[q]
        for i, j in enumerate(deltag_mean[::]):
            if i+1 >= len(deltag_mean):
                continue
            else:
                if deltag_mean[i+1] > 0.0 and deltag_mean[i] < 0.0 and deltag_mean[i+1] - deltag_mean[i] > 5e-3:
                    # print(deltag_med[i+1] - deltag_med[i])
                    # print(f"Changing index for q = {q:.3f} from {crossover_index} to {i}; -ve to +ve change.")
                    crossover_index = i+1
                elif deltag_mean[i+1] < 0.0 and deltag_mean[i] > 0.0 and np.abs(deltag_mean[i+1] - deltag_mean[i]) > 5e-3:
                    # print(deltag_med[i+1] - deltag_med[i])
                    # print(f"Changing index for q = {q:.3f} from {crossover_index} to {i}; +ve to -ve change.")
                    crossover_index = i+1
        # print(crossover_index)
        a=plt.plot(garray[::], deltag_mean, label=f"q = {q:.3f}")
        deltas[q] = deltag_mean
        # c=a[0].get_color()
        # plt.fill_between(g_s, 0.0, deltag_med, alpha=0.2, color=c)
        # plt.axvline(x=g_s[crossover_index], linestyle="--", alpha=0.5, color=c, label=f"q = {q:.3f}, g[{crossover_index}] = {g_s[crossover_index]:.3f}")

    plt.xticks(garray[::15])
    plt.grid(alpha=0.6)
    plt.ylabel(ylab)
    plt.xlabel(xlab)
    # plt.ylim((-0.01, 0.01))
    plt.title(f"{ylab} vs {xlab} for RG steps {step1},{step2} with {n} samples")
    plt.legend(loc = "upper right", bbox_to_anchor = (1.25, 1.0))
    if save:
        os.makedirs(f"{plotfolder}/plots", exist_ok=True)
        plt.savefig(f"{plotfolder}/plots/{var}_{metrictype}_RG{step1}-{step2}.png", dpi  = 150, bbox_inches="tight")
    return deltas
    # plt.show()

def convert_json_to_array(steps, jsondata, qarr, parr):
    ptot = np.empty(shape=(qarr.size, parr.size, steps, 2), dtype=np.float64)
    qtot = np.empty(shape=(qarr.size, parr.size, steps, 2), dtype=np.float64)
    for i,q in np.ndenumerate(qarr):
        gps = jsondata[f"{q}"]["p"]
        gqs = jsondata[f"{q}"]["q"]
        for step in range(steps):
            medp = gps[f"{step}"]["Median"]
            meanp = gps[f"{step}"]["Mean"]
            medq = gqs[f"{step}"]["Median"]
            meanq = gqs[f"{step}"]["Mean"]
            ptot[i, :, step, 0] = medp
            ptot[i, :, step, 1] = meanp
            qtot[i, :, step, 0] = medq
            qtot[i, :, step, 1] = meanq
    return ptot, qtot

def get_json_data(filename, first, excess, qsize, psize):
    """Helper function for using old local trial data"""
    step_1 = first
    step_2 = first+excess
    qs = np.linspace(0, 0.5, qsize)
    gs = np.linspace(0.01, 0.99, psize)
    gnum = gs.size
    with open(filename, "r") as file:
        data = json.load(file)
    pmed1, pmed2, pmean1, pmean2 = get_meds(step_1, step_2, data, qs, gs, "p", fromjson=True)
    qmed1, qmed2, qmean1, qmean2 = get_meds(step_1, step_2, data, qs, gs, "q", fromjson=True)

    # Convert metric dicts into np arrays
    pmets1 = np.empty(shape=(qs.size, gs.size, 2), dtype=np.float64)
    qmets1 = np.empty(shape=(qs.size, gs.size, 2), dtype=np.float64)
    pmets2 = np.empty(shape=(qs.size, gs.size, 2), dtype=np.float64)
    qmets2 = np.empty(shape=(qs.size, gs.size, 2), dtype=np.float64)

    for i, q in np.ndenumerate(qs):
        pmets1[i, :, :] = np.array([pmed1[q], pmean1[q]]).reshape((gnum,2))
        pmets2[i, :, :] = np.array([pmed2[q], pmean2[q]]).reshape((gnum,2))
        qmets1[i, :, :] = np.array([qmed1[q], qmean1[q]]).reshape((gnum,2))
        qmets2[i, :, :] = np.array([qmed2[q], qmean2[q]]).reshape((gnum,2))
    return pmets1, pmets2, qmets1, qmets2, qs, gs

def load_data(config: QSHEConfig, dataversion, wrong=False):
    """Load data from taskfarm"""
    pdata = np.load(f"{data_dir}/{dataversion}/QP/data/p_data_agg.npy")
    qdata = np.load(f"{data_dir}/{dataversion}/QP/data/q_data_agg.npy")
    pnan = np.isnan(pdata).sum()
    qnan = np.isnan(qdata).sum()
    if pnan > 0:
        print(f"{pnan} nan values found in pdata")
        pdata = np.nan_to_num(pdata, nan=0.0)
    if qnan > 0:
        print(f"{qnan} nan values found in qdata")
        qdata = np.nan_to_num(qdata, nan=0.0)
    # if wrong:
    #     pdata = 1 - (pdata/(1 - qdata))
    pvel = pdata[:, :, 1:, :] - pdata[:,:,:-1,:]
    qvel = qdata[:, :, 1:, :] - qdata[:,:,:-1,:]
    gs = np.linspace(config.p_min, config.p_max, config.p_num)
    qs = np.linspace(config.q_min, config.q_max, config.q_num)
    numsteps = config.steps
    return pvel, qvel, gs, qs, numsteps, pdata, qdata

def plot_stream(gs, qs, step_1, step_2, p_vel, q_vel, plotdir, contours, save, show, start = False, fixed = False):
    """Plot flow velocity streamplots"""
    # Setup grid and streamplot
    X, Y = np.meshgrid(gs, qs)
    # check for values where the velocity is ~ 0
    outfilename = f"{plotdir}/flow_diag"
    fixedtext = "fixed" if fixed else "not fixed"
    # tol = 1e-12
    # pmask = np.abs(p_vel) <= tol
    # qmask = np.abs(q_vel) <= tol
    # mask = np.logical_and(pmask, qmask)
    # dp_indices = np.argwhere(pmask)
    # dq_indices = np.argwhere(qmask)
    speed = np.sqrt(p_vel**2 + q_vel**2)
    # indices = np.argwhere(mask)
    fig = plt.figure(figsize=(12, 8))
    if start:
        start = np.array([X[::10], Y[::10]])
        plt.scatter(X[::10], Y[::10])
        stream = plt.streamplot(X, Y, p_vel, q_vel, color=speed, cmap="viridis", density=2.5, arrowsize=1.5, start_points=start.T)
    else:
        stream = plt.streamplot(X, Y, p_vel, q_vel, color=speed, cmap="viridis", density=1.5, arrowsize=1.5)
    fig.colorbar(stream.lines, label=r"$|\vec{v}_{pq}|$")
    plt.title(f"RG flow from step {step_1} to {step_2} for q {fixedtext}")
    handles = []
    labels = []
    # if indices.size > 0:
    #     vpq_handle = plt.scatter(gs[p_comb_indices], qs[q_comb_indices], linestyle="--", color="r", alpha=0.9)
    #     handles.append(vpq_handle)
    #     labels.append(r"$|\vec{v}_{pq}| = 0$")
    if contours:
        outfilename += "_contoured"
        pcontour= plt.contour(X, Y, p_vel, levels=[0.0], colors="m")
        qcontour = plt.contour(X, Y, q_vel, levels=[0.0], colors="k")
        plegend, _ = pcontour.legend_elements()
        qlegend, _ = qcontour.legend_elements()
        handles.append(plegend[0])
        handles.append(qlegend[0])
        labels.append(r"$\vec{v}_p = 0$")
        labels.append(r"$\vec{v}_q = 0$")
    if len(handles) > 0 and len(labels) > 0:
        plt.legend(handles=handles, labels= labels, loc = "upper right", bbox_to_anchor = (1.35, 1.0))
    # plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
    qbottom = qs.min() - 0.03
    qtop = qs.max() + 0.03
    plow = gs.min() - 0.03
    phigh = gs.max() + 0.03
    plt.xlim((plow, phigh))
    plt.ylim((qbottom, qtop))
    plt.xticks(np.round(np.linspace(gs.min(), gs.max(), 21), 2))
    plt.yticks(np.round(np.linspace(qs.min(), qs.max(), 21), 2))
    plt.xlabel(r"$p_{init}$")
    plt.ylabel(r"$q_{init}$")
    plt.grid(alpha=0.5)
    if save:
        # fig.savefig(f"C:/Users/ssark/Desktop/Uni/Year 4 Courses/Physics Final Year Project/QSHE Tests/2026-02-14/10000_samples/flow_diag_{step_1}to{step_2}.png", dpi=150)
        fig.savefig(f"{outfilename}_{step_1}to{step_2}.png", dpi=150)
    if show:
        plt.show()

def speed_heatmap(topindex, vel, velarray, gs, qs, plotdir, save = False):
    """Plot speed heatmaps"""
    # Heatmap of speed
    if vel == "p":
        veltitle = "\\vec{{v}}_{{p}}"
    elif vel == "q":
        veltitle = "\\vec{{v}}_{{q}}"
    else:
        veltitle = "|\\vec{{v}}_{{pq}}|"
    fig, ax = plt.subplots(3,1,figsize=(6, 12), sharex=True, sharey=True)
    m = ax[0].imshow(velarray[:,:,topindex, 0], cmap="Wistia", extent=[gs.min(), gs.max(), qs.min(), qs.max()], origin="lower")
    ax[0].set_title(f"Heatmap of ${veltitle}$ for RG step {topindex}-{topindex+1}")
    ax[1].imshow(velarray[:,:,topindex-1, 0], cmap="Wistia", extent=[gs.min(), gs.max(), qs.min(), qs.max()], origin="lower")
    ax[1].set_ylabel("q")
    ax[1].set_title(f"Heatmap of ${veltitle}$ for RG step {topindex-1}-{topindex}")
    ax[2].imshow(velarray[:,:,topindex-2, 0], cmap="Wistia", extent=[gs.min(), gs.max(), qs.min(), qs.max()], origin="lower")
    ax[2].set_xticks(ticks=np.round(gs[::50], 2))
    ax[2].set_yticks(ticks = qs[::10])
    ax[2].set_xlabel("p")
    ax[2].set_title(f"Heatmap of ${veltitle}$ for RG step {topindex-2}-{topindex-1}")
    for a in ax:
        a.grid(alpha=0.4)
    fig.colorbar(m, ax=ax, label=f"${veltitle}$")
    # fig.tight_layout()
    os.makedirs(f"{plotdir}/vfield/{vel}", exist_ok=True)
    if save:
        fig.savefig(f"{plotdir}/vfield/{vel}/{vel}vel_heatmap_steps{topindex-2}to{topindex+1}.png", dpi=150)
    else:
        plt.show()

def find_peaks(stepnum, pdata, qdata, del_speed, speedtol):
    """Find candidate fixed points from a given RG step, using a tolerance on the flow speed"""
    # Check which points possess the lowest speed for candidate FPs
    # print(del_speed.shape)
    stepnum = 2
    vn = del_speed[:, :, -stepnum, 0]
    pn = pdata[:, :, -stepnum, 0]
    qn = qdata[:, :, -stepnum, 0]
    # print(f.shape)
    speed_tol = np.percentile(vn, speedtol)
    velmask = vn <= speed_tol
    pfp = pn[velmask]
    qfp = qn[velmask]
    # print(pfp.shape)
    print(f"Num fp (p) = {pfp.size}\nRange = {pfp.min():.7f} to {pfp.max():.7f}")
    print(f"Num fp (q) = {qfp.size}\nRange = {qfp.min():.7f} to {qfp.max():.7f}")
    # Plot these points
    pbins, qbins = 100, 100
    h2d, qedge, pedge = np.histogram2d(qfp, pfp, bins=[qbins, pbins])

    # Simple peak finder
    flattened_indexes = np.argsort(h2d.ravel())[::-1]
    # top_percent = 5
    peaks = []
    for index in flattened_indexes[:]:
        qind, pind = np.unravel_index(index, h2d.shape) # Original p,q indexes
        if h2d[qind, pind] == 0: # Skip if its the origin
            continue
        pcenter = 0.5 * (pedge[pind] + pedge[pind+1])
        qcenter = 0.5 * (qedge[qind] + qedge[qind+1])
        peaks.append((pcenter, qcenter, h2d[qind, pind]))
    return peaks

# Compute the jacobian from our discrete RG flow
def get_jacobian_and_eigenvals(n, step, rgsteps, p, q, gs, qs, ind1, ind2, peak_p, peak_q):
    """Compute the jacobian matrix between 2 consecutive RG steps for a pair of indices, and return their eigenvalues"""
    m = n + step
    if m >= rgsteps:
        raise ValueError(f"Cannot exceed number of RG steps {n} + {step} >= {rgsteps}")
    p_n = p[:, :, n, 0]
    q_n = q[:, :, n, 0]
    p_m = p[:, :, m, 0]
    q_m = q[:, :, m, 0]

    dp = gs[1] - gs[0]
    dq = qs[1] - qs[0]
    i, j = get_peaks(p_n, q_n, peak_p, peak_q)
    ind1 = i
    ind2 = j
    dpm_dqn, dpm_dpn = np.gradient(p_m, dq, dp)
    dqm_dqn, dqm_dpn = np.gradient(q_m, dq, dp)
    jacob = np.array([[dpm_dpn[ind1, ind2], dpm_dqn[ind1,ind2]],
                                [dqm_dpn[ind1,ind2], dqm_dqn[ind1, ind2]]])
    eigenvals = np.linalg.eigvals(jacob)
    return eigenvals

# Now we need to compute the eigenvalues at the candidate peaks
def get_peaks(p_n, q_n, target_p, target_q):
    """Convert the candidate peak value to the nearest index in the histogram"""
    square_distance = (p_n - target_p)**2 + (q_n - target_q)**2
    index = np.argmin(square_distance)
    return np.unravel_index(index, p_n.shape)

# Set phase labels
def get_boundary(pdata, qdata, lastindex, lb = 0.2, ub = 0.8):
    plast = pdata[:, :, lastindex, 0]
    # qlast = qdata[:, :, lastindex, 0]

    # Arbitrary choices, can be modified
    leftphase = plast < lb
    rightphase = plast > ub
    # neitherphase = ~(leftphase | rightphase) # in between

    # Labels - let the left phase be 0, right phase be 1.
    labels = np.full(plast.shape, 2, dtype=np.int8)
    labels[leftphase] = 0
    labels[rightphase] = 1

    # Check where the labels change
    pswap = labels[:, 1:] != labels[:, :-1]
    qswap = labels[1:, :] != labels[:-1, :]

    bound = np.zeros_like(labels, dtype=bool)
    bound[:, 1:] |= pswap
    bound[1:, :] |= qswap
    return bound

# We need to order the data points before we can fit a line to them
def nearest_neighbour_ordering(p, q):
    pqgrid = np.column_stack([p, q])
    dim = pqgrid.shape[0]

    # Origin at bottom left
    starting = np.argmin(pqgrid[:, 0])
    passed = np.zeros(dim, dtype=bool)
    ordered_grid = np.empty(dim, dtype=int)

    ordered_grid[0] = starting
    passed[starting] = True
    for i in range(1, dim):
        ending = pqgrid[ordered_grid[i-1]]
        # Distance to unused points
        dist = pqgrid[~passed] - ending
        sqdist = np.sum(dist**2, axis=1)
        nextval = np.flatnonzero(~passed)[np.argmin(sqdist)]
        ordered_grid[i] = nextval
        passed[nextval] = True

    return ordered_grid

# Then we can fit a line
def make_smooth_line(p, q, s, k, out):
    ordered_points = nearest_neighbour_ordering(p, q)
    pordered = p[ordered_points]
    qordered = q[ordered_points]

    # Remove duplicates
    stacked = np.column_stack([pordered, qordered])
    _, unique = np.unique(stacked, axis=0, return_index=True)
    unique_sorted = np.sort(unique)
    psorted = pordered[unique_sorted]
    qsorted = qordered[unique_sorted]

    # Fit line
    spl, u = make_splprep([psorted, qsorted], s=s, k=min(k, len(psorted)-1))
    new_u = np.linspace(0, 1, out)
    pfit, qfit = spl.__call__(new_u)
    return np.asarray(pfit), np.asarray(qfit)

def plot_boundaries(colormaps, pdata, qdata, numsteps, gs, qs, lb, ub, plotdir, plotsmooth = True, save = False, s = 1e-1, k = 3, out = 600):
    """
    Plot boundaries for input data. Boundary lines represent regions where enclosed initial system configs have arrived below the specified boundaries
    Eg: if lb = 0.2 and ub = 0.8, the line drawn for step k=4 encloses the region of initial configs that have not arrived below 0.2 or above 0.8 by step k
    Note: lb, ub track the value of p
    """
    bounds = []
    # gridx, gridy = np.meshgrid(gs, qs)
    # xsmooth = np.linspace(0, 1.0, 500)
    plt.figure(figsize=(12,8))
    # lb=0.2
    # ub=0.8
    for i in range(0,numsteps,2):
        b = get_boundary(pdata, qdata, i, lb, ub)
        bounds.append(b)
        y, x = np.where(b==True)
        if plotsmooth:
            p, q = make_smooth_line(gs[x], qs[y], s, k, out)
            plt.plot(p, q, alpha=0.6)
        plt.scatter(gs[x][::1], qs[y][::1], alpha=0.7, label=f"Step {i}")
        # plt.imshow(b, cmap=colormaps[i], interpolation="nearest", extent=[gs.min(), gs.max(), qs.min(), qs.max()], origin="lower")

    for i in range(1,len(bounds)):
        diff = np.mean(bounds[i] == bounds[i-1])
        print(f"{i}, {diff}")
    plt.legend(loc="upper right", bbox_to_anchor=(1.15, 1.0))
    plt.grid(alpha=0.5)
    plt.axvline(0.0, linestyle="--", alpha=0.4, color="g")
    plt.axvline(1.0, linestyle="--", alpha=0.4, color="g")
    plt.axhline(np.min(qs), linestyle="--", color="r", alpha=0.4)
    plt.axhline(np.max(qs), linestyle="--", color="r", alpha=0.4)
    plt.xticks(np.linspace(0.0, 1.0, 21))
    plt.yticks(np.linspace(np.min(qs), np.max(qs), 21))
    plt.xlabel("p")
    plt.ylabel("q")
    # plt.ylim((0.0, 0.5))
    plt.title("Boundaries across RG flow")
    if save:
        plt.savefig(f"{plotdir}/boundary/boundaries_{lb}to{ub}.png", dpi=150, bbox_inches="tight")
    else:
        plt.show()

def plot_densities(binnum, datarange, varname, vardata, numsteps, plotdir, density = True, save = False):
    """Make histograms of the input variable data"""
    plt.figure(figsize=(12, 8))
    zbin = binnum
    for i in range(numsteps):
        zhist, zedges = np.histogram(vardata[:,:, i, 1], bins=zbin, range=datarange, density=density)
        plt.plot(0.5*(zedges[1:]+zedges[:-1]), zhist, label=f"Step {i}")
        maxindex = np.argmax(zhist)
        maxedge = zedges[maxindex+1]
        maxcenter = 0.5 * (zedges[maxindex] + zedges[maxindex+1])
        # print(maxcenter)
        plt.scatter(maxcenter, zhist[maxindex])
        # print(np.median(zhist))
    plt.xlabel(f"{varname}")
    plt.ylabel(f"P({varname})")
    plt.title(f"Density histogram of {varname}")
    plt.legend(loc="upper right", bbox_to_anchor=(1.15, 1.0))
    if save:
        plt.savefig(f"{plotdir}/density/{varname}_density_{zbin}bins.png", dpi=150, bbox_inches="tight")
    else:
        plt.show()

def plot_gammas(qviewindex, numsteps, gammas, gs, qs, plotdir, start_step, endstep, save = True):
    """Plot Gamma_n(p) graphs for each q value"""
    plt.figure(figsize=(9,6))
    if endstep > numsteps:
        endstep = numsteps
    for k in range(start_step, endstep):
        plt.plot(gs, gammas[qviewindex, :, k], label=f"Step {k}")
    plt.grid(alpha=0.4)
    plt.xlabel("p")
    plt.ylabel(r"$\Gamma_{n}(\text{p})$")
    plt.title(f"$\\Gamma_{{n}}(\\text{{p}})$ vs p for q = {qs[qviewindex]:.3f}")
    # plt.xlim((0.3, 0.7))
    # plt.ylim((-0.05, 0.3))
    # plt.xticks(np.linspace(0.0, 1.0, 21))
    plt.legend(loc="upper right", bbox_to_anchor=(1.15, 1.0))
    if save:
        plt.savefig(f"{plotdir}/gamma_q_{qs[qviewindex]:.3f}_step{start_step}to{endstep-1}.png", dpi=150, bbox_inches="tight")
    else:
        plt.show()

# Given z arrays for each q, we need to find the intersection p value
def crossing_p(p, y1, y2):
    """
    Find p where y1(p)-y2(p)=0 by linear interpolation between sign changes.
    Returns np.nan if no crossing.
    """
    d = y1 - y2
    s = np.sign(d)
    idx = np.flatnonzero(s[1:] * s[:-1] < 0)
    if idx.size == 0:
        return np.nan
    j = idx[0]  # first crossing
    p1, p2 = p[j], p[j+1]
    d1, d2 = d[j], d[j+1]
    return p1 - d1*(p2-p1)/(d2-d1)

def plot_crossings(gammas, gs, qs, numsteps, start, end, pstart, pend, plotdir, save = False, show = False):
    """Plot p_c(q) graphs"""
    # Plot the crossings for each q
    if end > numsteps:
        end = numsteps
    plt.figure(figsize=(10, 6))
    crossings = np.empty(shape=(qs.size, end-start-1), dtype=np.float64)
    for steps in range(start, end-1):
        # steps = 5
        pc = np.zeros(shape=(qs.size, steps-1), dtype=np.float64)
        startstep = end - steps
        indstep = 0
        for s1 in range(startstep, end-1):
            s2 = s1+1
            for iq in range(qs.size):
                if iq == 0:
                    pc[iq,indstep] = crossing_p(gs[200:300], gammas[iq,200:300,s1], gammas[iq,200:300,s2])
                else:
                    pc[iq,indstep] = crossing_p(gs[pstart:pend], gammas[iq,pstart:pend,s1], gammas[iq,pstart:pend,s2])
            indstep += 1
        plt.plot(np.mean(pc, axis=1), qs, label=f"avg of {end-steps} to {end-1}")
        plt.plot(pc[:, 0], qs, label=f"steps = {end-steps} to {end-steps+1}", linestyle="--")
        crossings[:, steps-start] = pc[:, 0]
    plt.legend(loc="upper right", bbox_to_anchor=(1.21, 1.0))
    plt.title(r"$q_{init}$ vs $\bar{p}_c$")
    plt.ylabel(r"$q_{init}$")
    plt.xlabel(r"$p_c$")
    if save:
        plt.savefig(f"{plotdir}/Gamma/crossings_step{start}to{end-1}.png", dpi=150, bbox_inches="tight")
        return crossings
    if show:
        plt.show()
        return crossings
    return crossings

def gamma_slope(ps, gamma_data, p_c, k = 5, degree = 2):
    """ Compute the slope dGamma/dp at p_c for a particular q value and step. Assumes gamma_data is of shape (ps.size,)"""
    # nearest p to p_c
    nearest_index = np.argmin(np.abs(ps - p_c))

    # Set a 2k window of values around p_c for fitting the polynomial
    start_index = min(0, nearest_index - k)
    end_index = max(ps.size-1, nearest_index+k+1)
    pvals = ps[start_index:end_index]
    gammavals = gamma_data[start_index:end_index]
    pcenter = pvals - p_c

    # Guard against not setting enough data points for the chosen degree
    eff_degree = min(degree, len(pcenter) - 1)
    if eff_degree < 1:
        return np.nan

    coefs = np.polyfit(pcenter, gammavals, eff_degree)
    pred = np.polyval(coefs, pcenter)
    ymean = np.mean(gammavals)
    res = np.sum((gammavals - pred)**2)
    tot = np.sum((gammavals - ymean)**2)
    errs = np.sqrt(np.sum(res**2)/(pred.size))
    r2 = 1 - (res/tot)
    # print(f"R2 = {r2}")
    # backwards index to get linear coefficient regardless of degree
    slope = coefs[-2]
    return slope, r2, errs

def gamma_neighbour_slope():
    pass

def fit_nu(ks, slope_array, starting_k):
    """Use a least squares fit for ln(T_k) = k*ln(2)/nu"""
    mask = (ks >= starting_k) & np.isfinite(slope_array)
    # print(mask)
    # print((slope_array < 0).sum())
    use_k = ks[mask]
    # print(use_k)
    # print(slope_array[mask])
    # Assumes input slope array is per q
    y = np.log(np.abs(slope_array[mask]))
    # print(y)
    # print((slope_array < 0).sum())
    # If there are too few steps to use
    if use_k.size < 2:
        return np.nan, np.nan, np.nan

    # a = np.column_stack([use_k, np.ones_like(use_k)])
    a = 2 ** (use_k+1)
    m, b = np.polyfit(np.log(a), y, 1)
    # print(m)
    # nu = np.log(2) / m
    nu = 1 / m
    return nu, m, b


# Plot coordinates for two consecutive RG steps
def grid_coords(ps, qs, pdata, qdata, datastep, step1, step2, plotdir, save=False, all=False):
    p0 = pdata[:, :, step1, 0]
    p1 = pdata[:, :, step2, 0]
    q0 = qdata[:, :, step1, 0]
    q1 = qdata[:, :, step2, 0]
    # x, y = np.meshgrid(p0, q0)
    # pv = p1 - p0
    # qv = q1 - q0
    pstep = datastep
    qstep = datastep//5
    x0 = p0[::qstep, ::pstep].ravel()
    x1 = p1[::qstep, ::pstep].ravel()
    y0 = q0[::qstep, ::pstep].ravel()
    y1 = q1[::qstep, ::pstep].ravel()
    # print(x0.shape, x1.shape)
    # print(p0.shape, q0.shape)
    pvcoords = x1-x0
    qvcoords = y1-y0
    # qstep = qs.size // 10
    qlims = (-0.02, qs.max()+0.02)
    # pstep = ps.size // 10
    plims = (-0.02, 1.0+0.02)
    pticks = np.round(np.linspace(0.0, 1.0, 11), 2)
    qticks = np.round(np.linspace(0.0, 0.5, 11), 2)
    grid1, grid2 = np.meshgrid(ps[::pstep], qs[::qstep])
    xg = grid1.ravel()
    yg = grid2.ravel()
    if step1 == 0:
        fig, ax = plt.subplots(figsize=(9, 6))
        if all:
            ax.scatter(xg, yg, color="k", s=20, alpha=1.0, zorder=2,label="Init")
            ax.quiver(xg, yg, x1-xg, y1-yg, zorder=1, angles="xy", scale_units="xy", scale=1, width=0.003, color="g", alpha=0.5)
            ax.scatter(x1, y1, color="r", s=20, alpha=1.0, zorder=2,label=f"{step2}")
            title=f"step {step2}"
        else:
            ax.scatter(xg, yg, color="k", s=20, alpha=1.0, label="Init")
            ax.quiver(xg, yg, x0-xg, y0-yg, zorder=1, angles="xy", scale_units="xy", scale=1, width=0.003, color="g", alpha=0.5)
            ax.scatter(x0, y0, color="r", s=20, alpha=1.0, label="Start")
            title="Start"
        ax.scatter(0.5, 0.0, color="b", marker="*", s=100, alpha=1.0, zorder=3)
        # print(np.max(x0 - (xg + (x0-xg))))
        ax.set_xlabel(r"$p$", fontsize=20)
        ax.set_ylabel(r"$q$", fontsize=20)
        ax.tick_params(labelsize=16)
        ax.minorticks_on()
        # ax.set_title(f"RG flow from init to {title}")
        ax.grid(alpha=0.5)
        ax.set_xticks(pticks)
        ax.set_yticks(qticks)
        # print(xgrid.min())
        ax.set_xlim(plims)
        ax.set_ylim(qlims)
        # ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1.0))
        plt.tight_layout()
        if save:
            fig.savefig(f"{plotdir}/grid/flow_grid_Initto{step1}.png", dpi=150, bbox_inches="tight")
        else:
            plt.show()
        close_old_plots()
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(x0, y0, s=20, color="k", zorder=2, label="Start")
        ax.quiver(x0, y0, pvcoords, qvcoords, angles="xy", scale_units="xy", scale=1, zorder=1, width=0.003, color="g", alpha=0.5)
        ax.scatter(x1, y1, s=20, color="r", alpha=1.0, zorder=2,label="End")
        ax.scatter(0.5, 0.0, color="b", marker="*", s=100, alpha=1.0, zorder=3)
        print(np.max(y0), np.max(y1))
        # print(np.max(x0 - (xg + (x0-xg))))
        ax.set_xlabel(r"$p$", fontsize=20)
        ax.set_ylabel(r"$q$", fontsize=20)
        # ax.set_title(f"RG flow from step {step1} to {step2}")
        ax.grid(alpha=0.5)
        ax.set_xticks(pticks)
        ax.set_yticks(qticks)
        ax.tick_params(labelsize=16)
        ax.minorticks_on()
        # print(xgrid.min())
        ax.set_xlim(plims)
        ax.set_ylim(qlims)
        # ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1.0))
        plt.tight_layout()
        if save:
            fig.savefig(f"{plotdir}/grid/flow_grid_{step1}to{step2}.png", dpi=150, bbox_inches="tight")
        else:
            plt.show()
    else:
        # grid1, grid2 = np.meshgrid(x0, y0)
        # ax.quiver(grid1, grid2, x0-grid1.ravel(), y0-grid2.ravel(), zorder=4, label="Init to start")
        # print(x0.size, y0.size)
        # ax.scatter((x1-x0)[::10], (y1-y0)[::10], label="test")
        # print(p0, q0)
        # print(np.mean(x1-x0), np.mean(y1-y0))
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(x0, y0, s=20, color="k", zorder=2, label="Start")
        ax.quiver(x0, y0, pvcoords, qvcoords, angles="xy", scale_units="xy", scale=1, zorder=1, width=0.003, color="g", alpha=0.5)
        ax.scatter(x1, y1, s=20, color="r", alpha=1.0, zorder=2,label="End")
        ax.scatter(0.5, 0.0, color="b", marker="*", s=100, alpha=1.0, zorder=3)
        ax.set_xlabel(r"$p$", fontsize=20)
        ax.set_ylabel(r"$q$", fontsize=20)
        # ax.set_title(f"RG flow from step {step1} to {step2}")
        ax.grid(alpha=0.5)
        ax.set_xticks(pticks)
        ax.set_yticks(qticks)
        ax.tick_params(labelsize=16)
        ax.minorticks_on()
        # print(xgrid.min())
        ax.set_xlim(plims)
        ax.set_ylim(qlims)
        # ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1.0))
        plt.tight_layout()
        if save:
            fig.savefig(f"{plotdir}/grid/flow_grid_{step1}to{step2}.png", dpi=150, bbox_inches="tight")
        else:
            plt.show()


# ## Post-Taskfarm implementation

# In[3]:


# Load data for the desired vars
# z_moments = defaultdict(list)
# g_moments = defaultdict(list)
# for val in theta_vals:
#     z_moments[f"{val}"] = load_var_moments(12, "z", val)
#     g_moments[f"{val}"] = load_var_moments(12, "g", val)

# Primers for older tests, potential use later
markers = [",", "o", "v", "^", "1", "2", "3", "4", "8", "s", "p", "P", "*", "+"]
loaded_colors = mcolors.TABLEAU_COLORS
avail_colors = len(loaded_colors)
avail_markers = len(markers)
close_old_plots() # Close old plots for a quick reset when desired


# ## Taskfarm data analysis

# In[4]:


# For old local data
# 10k samples, not fixed q
# pmets1, pmets2, qmets1, qmets2, qs, gs = get_json_data("C:/Users/ssark/Desktop/Uni/Year 4 Courses/Physics Final Year Project/QSHE Tests/2026-02-14/10000_samples/metrics_10000samples_6steps.json", 0, 1, 21, 200)
# 50k samples, fixed q
# pmets1, pmets2, qmets1, qmets2, qs, gs = get_json_data("C:/Users/ssark/Desktop/Uni/Year 4 Courses/Physics Final Year Project/QSHE Tests/2026-02-14/50000_samples/metrics_50000samples_8steps.json", 0, 1, 21, 100)

# Use new trial data from taskfarm
dataversion = "qp_unfixed_numerical_shreyas"
dataconfig = load_yaml(f"{data_dir}/{dataversion}/QP/config/updated_config.yaml")
plotdir = f"{data_dir}/{dataversion}/QP/plots"
extraplotfolders = ["boundary", "density", "Gamma", "grid", "Nu", "vfield"]

os.makedirs(plotdir, exist_ok=True)
os.makedirs(f"{data_dir}/{dataversion}/QP/output", exist_ok=True)
for e in extraplotfolders:
    os.makedirs(f"{plotdir}/{e}", exist_ok=True)
cfg = build_config(dataconfig)
datafixed = bool(cfg.fixed)
wrong = False
pvel, qvel, gs, qs, numsteps, pdata, qdata = load_data(cfg, dataversion, wrong)
# Compute all derived quantities
gdata = pdata + (1 - pdata)*qdata
gdata = np.clip(gdata, 1e-9, 1-1e-9)
zdata = convert_g_to_z(gdata)
zdata = zdata[:, :, :, :]
tdata = np.sqrt(pdata)
fdata = np.sqrt(qdata*(1 - pdata))
vardatadict = {"t": tdata, "f": fdata, "p": pdata, "q": qdata, "g": gdata, "z": zdata}


# In[6]:


pstd = pdata[:, :, :, 2]
qstd = qdata[:, :, :, 2]

ps = np.linspace(cfg.p_min, cfg.p_max, cfg.p_num)
qst = np.linspace(cfg.q_min, cfg.q_max, cfg.q_num)
# qchoice = 0
# pchoice = 249
os.makedirs(f"{plotdir}/stds", exist_ok=True)
os.makedirs(f"{plotdir}/variance", exist_ok=True)
plotvars = ["std", "variance"]
stddone = True
vardone = False
def plot_mets(pstd, qstd, ps, qst, cfg, variable = "std", save=True):
    for qchoice in range(qst.size):
        fig, ax = plt.subplots(1, 2, figsize=(12, 8))
        pchoice = min(qchoice * 10, ps.size)
        if variable == "std":
            pdat = pstd
            title = "$\\sigma_p$"
            f = "stds"
        else:
            pdat = pstd**2
            title = "$\\sigma^2_p$"
            f = "variance"
        for i in range(cfg.steps):
            ax[0].plot(ps, pdat[qchoice, :, i])
            ax[1].plot(qst, pdat[:, pchoice, i], label=f"RG Step {i}")
        plt.suptitle(f"{title} for $q_{{init}} = {qst[qchoice]:.3f}$ and $p_{{init}} = {ps[pchoice]:.3f}$")
        ax[0].set_ylabel(f"{title}")
        ax[0].set_title(f"{title}$(p_{{init}})$")
        ax[1].set_title(f"{title}$(q_{{init}})$")
        # ax[0].legend(bbox_to_anchor=(1.2, 1.0))
        ax[1].legend(bbox_to_anchor=(1.05, 1.0))
        if save:
            plt.savefig(f"{plotdir}/{f}/q{qst[qchoice]:.3f}_p{ps[pchoice]:.3f}.png", dpi=150, bbox_inches="tight")
        if qchoice % 10 == 0:
            close_old_plots()
    close_old_plots()
# plt.show()
# plot_mets(pstd, qstd, ps, qst, cfg, "std", True)


# ### Field analysis

# #### Streamplots

# In[ ]:


excess = 1
make_contours = False
save_plots = True
show_plots= False
close_old_plots()
# Before further investigation, make all plots once
streamplotdir = f"{plotdir}/vfield"
for i in range(numsteps-1):
    j = i+1
    # Compute the velocity arrays
    p_vel = pvel[:, :, i, 0]
    q_vel = qvel[:, :, i, 0]
    plot_stream(gs, qs, i, j, p_vel, q_vel, streamplotdir, make_contours, save_plots, show_plots, fixed=datafixed)
close_old_plots()


# In[75]:


step_1 = 7
step_2 = step_1 + excess
# Compute the velocity arrays
p_vel = pvel[:, :, step_1, 0]
q_vel = qvel[:, :, step_1, 0]
pq_speed = np.sqrt(p_vel**2 + q_vel**2)


# In[76]:


# For individual viewing
close_old_plots()
plot_stream(gs, qs, 3, 4, p_vel, q_vel, streamplotdir, contours=False, save=False, show=True, fixed=datafixed)


# ### Check grid flow

# In[217]:


gridstep = 14
for i in range(0, numsteps-1, gridstep):
    close_old_plots()
    if i + gridstep < numsteps:
        grid_coords(gs, qs, pdata, qdata, 20, i, i+gridstep, plotdir, True, False)


# ### Checking phases

# #### Arbitrary bounds check

# In[5]:


close_old_plots()
# Take the final RG step data for p and q
# p_all, q_all = convert_json_to_array(numsteps, data, qs, gs)
p_all = pdata
q_all = qdata
p_fin = p_all[:, :, numsteps-1, :]
q_fin = q_all[:, :, numsteps-1, :]
# Store the per-step change
del_p = np.empty(shape=(qs.size, gs.size, numsteps-1, 3), dtype=np.float64)
del_q = np.empty(shape=(qs.size, gs.size, numsteps-1, 3), dtype=np.float64)
for s in range(1,numsteps):
    del_p[:, :, s-1, :] = p_all[:,:,s,:] - p_all[:, :, s-1,:]
    del_q[:, :, s-1, :] = q_all[:,:,s,:] - q_all[:,:,s-1,:]

del_speed = np.sqrt(del_p**2 + del_q**2)
# compute overall paths
path_p = np.sum(del_p, axis=2)
path_q = np.sum(del_q, axis=2)
arc_pq = np.sum(del_speed, axis=2)

# Find out which ps end up around 0 or 1
p_ins0 = np.abs(p_fin[:,:,0]) <= 1e-6
p_ins1 = p_fin[:,:,0] >= 1.0 - 2e-6
# Get their indices
q0_indices, p0_indices = np.where(p_ins0)
q1_indices, p1_indices = np.where(p_ins1)
plt.xlabel("p")
plt.ylabel("q")
plt.title(f"$p_{{med}}$ flow regimes of the system")
plt.scatter(gs[p0_indices], qs[q0_indices], label="Low", color="g")
plt.scatter(gs[p1_indices], qs[q1_indices], color="m", label="High")
plt.legend(bbox_to_anchor=(1.2, 1.0))
plt.show()


# #### Velocity heatmap

# In[79]:


close_old_plots()
# topindex = numsteps-2
vellabels = ["p", "q", "pq"]
speedarrs = {"p": del_p, "q": del_q, "pq": del_speed}
saveheatmap = True
for topindex in range(2, numsteps-2):
    for label in vellabels:
        speed_heatmap(topindex, label, speedarrs[label], gs, qs, plotdir, saveheatmap)
        close_old_plots()


# #### Check candidate FPs and eigenvalues

# In[80]:


stepchoice = 2
speedtol = 1.0
peaks = find_peaks(stepchoice, pdata, qdata, del_speed, speedtol)
print(len(peaks))
for i in range(0,len(peaks)):
    print(get_jacobian_and_eigenvals(0, 1, numsteps, pdata, qdata, gs, qs, 1,1, peaks[i][0], peaks[i][1]))


# #### Boundary plots

# In[122]:


close_old_plots()
# lastindex = 0
cmaps = ["viridis", "Wistia", "autumn", "winter", "summer"]
lower = 0.2
upper = 0.8
plotfit = False
saveboundary = True
plot_boundaries(cmaps, pdata, qdata, numsteps, gs, qs, lower, upper, plotdir, plotfit, saveboundary, s=1e-2)


# #### Density Plots

# In[82]:


varchoices = list(vardatadict.keys())
for varchoice in varchoices:
    if varchoice == "z":
        varrange = (-5.0, 5.0)
        varbins = 200
    else:
        varbins = 50
        varrange = XLIMS["FP"][varchoice]
    # plt.ioff()
    plot_densities(varbins, varrange, varchoice, vardatadict[varchoice], numsteps, plotdir, save=True)
    close_old_plots()


# ### Gamma Analysis

# In[6]:


# A reproduction of kobayashi's gamma(p) idea
# Our z is ~ system size / Xi.
# print(qs.size)
gammas = np.empty(shape=(qs.size, gs.size, numsteps), dtype=np.float64)
for j in range(qs.size):
    for k in range(numsteps):
        zn = zdata[j, :, k, 0]
        system_size = 2**(k+1)
        gamma = zn / system_size
        # gamma = zn/2
        gammas[j, :, k] = gamma
# Plot all gammas once
savegamma = False
gamma_start = 2
gamma_end = 14
# f = f"{plotdir}/Gamma/{gamma_start}to{gamma_end-1}"
# os.makedirs(f, exist_ok = True)
# for plotq in range(qs.size):
# plot_gammas(25, numsteps, zdata[:, :, :, 0], gs, qs, f, gamma_start, gamma_end, savegamma)
plt.figure(figsize=(10, 8))
plt.xlabel(r"$p$", fontsize=20)
plt.ylabel(r"$z$", fontsize=20)
plt.xlim((0.8, 0.9))
plt.ylim((-4, 2))
for i in range(gamma_start, gamma_end)[::1]:
    plt.plot(gs, zdata[10, :, i, 0])
    # plt.plot(gs, zdata[25, :, i, 0], linestyle = "--")
    plt.plot(gs, zdata[30, :, i, 0], linestyle="-.")
# plt.axhline(-1.79)
pc = fits[:, :-1]
pcs = np.mean(pc, axis=1)
# print(pcs)
print(qgamma[30])
plt.xticks(np.linspace(0.8, 0.90, 11), fontsize=14)
plt.yticks(list(np.linspace(-4, 2, 7))+[-1.79], [str(k) for k in list(np.round(np.linspace(-4, 2, 7), 3))]+[-1.79], fontsize=14)
plt.scatter(pcs[10]*0.999, -1.79, marker = "*", color="k", zorder=2, s=100, label=r"$p_c(0.102)$")
plt.scatter(pcs[30], -1.79, marker="*", color="k", s=100, zorder=2, label=r"$p_c(0.306)$")
# plt.scatter(pcs[10:40], np.array([-1.79]*30), marker="*", color="m", zorder=2)
plt.axhline(-1.79, linestyle="--", color="k", alpha=0.5)
plt.ylim((-4.05, 2.05))
plt.xlim((0.795, 0.905))
plt.grid(alpha=0.5)
plt.minorticks_on()
plt.legend(loc="upper right", fontsize=20)
# plt.savefig("./report/z_pc.pdf", dpi=150, bbox_inches="tight")
plt.show()
close_old_plots()


# ### Useleess

# In[8]:


# Individual viewing
plt.ion()
plotq = 0
plotp = 249 +1
start = 1
end = 12
# plot_gammas(plotq, numsteps, zdata[:, :, :, 0], gs[:], qs, f, start, end, False)
close_old_plots()
# print(zdata[plotq, plotp, :, 0])
from scipy.stats import norm
# z = norm.pdf(range(1,numsteps+1), loc=zdata[plotq, plotp, :, 1], scale = zdata[plotq, plotp, :, 2])
# print(z.shape)
xs = range(start, end)
fitmeans = np.empty((len(xs)))
fitstds = np.empty((len(xs)))
for i in range(len(xs)):
    x = np.linspace(zdata[plotq, plotp, i, 0] - 2.5*zdata[plotq, plotp, i, 2], zdata[plotq, plotp, i, 0] + 2.5*zdata[plotq, plotp, i, 2], 200) # Plot the data simply over a range of 4 STDs
    print(x.min(), x.max())
    fitmean, fitstd = norm.fit(x, loc=zdata[plotq, plotp, i, 0], scale=zdata[plotq, plotp, i, 2])
    fitmeans[i] = fitmean
    fitstds[i] = fitstd
    # print(fitparams)
    fittedz = norm.pdf(x, loc=fitmean, scale=fitstd)
    unfitz = norm.pdf(x, loc = zdata[plotq, plotp, i, 0], scale = zdata[plotq, plotp, i, 2])
    # print(zdata[plotq, plotp, i, 0], zdata[plotq, plotp, i, 1], zdata[plotq, plotp, i, 2])
    # print(fitmean, fitstd)
plt.scatter(range(start, end), fitmeans, label="Fitted mean", marker="*", color="g", s = 200)
plt.scatter(range(start, end), zdata[plotq, plotp, start:end, 0], label="Median", marker="+", color="r", s=100)
plt.scatter(range(start, end), zdata[plotq, plotp, start:end, 1], label="Mean", marker="o", color="b")
plt.scatter(range(start, end), fitmeans - zdata[plotq, plotp-1, start:end, 0], label="Gap w/ -1")
plt.scatter(range(start, end), fitmeans - zdata[plotq, plotp-2, start:end, 0], label="Gap w/ -2")
plt.scatter(range(start, end), zdata[plotq, plotp-1, start:end, 0], label="-1")
plt.scatter(range(start, end), zdata[plotq, plotp-2, start:end, 0], label="-2")
plt.xticks(xs)
plt.xlabel("RG steps")
plt.ylabel(r"$\bar{z}$")
plt.title(f"$\\bar{{z}}$ vs RG steps for q = {qs[plotq]:.3f} and p = {gs[plotp]:.3f}")
    # plt.plot(x[::10], fittedz[::10], label=f"Fitted {i}")
    # plt.plot(x[::10], unfitz[::10], label=f"Step {i}", linestyle="--")
    # plt.plot(gs[::10], z[::10, i], label=f"Step {i}")
    # plt.plot(gs[::10], fittedz[::10], linestyle="--", label=f"Fitted {i}")
    # plt.plot(x[::10], z[::10, i], label=f"Step {i}")
    # plt.plot(x[::10], fittedz[::10], linestyle="--", label=f"Fitted {i}")
plt.legend(loc="upper right", bbox_to_anchor=(1.30, 1.0))
plt.show()


# ### Important

# In[7]:


# Restrict the range of our analysis to where the crossing mainly lies
val_region = np.where(np.logical_and(gs > 0.3, gs < 0.95))
val_start = np.min(val_region)
val_end = np.max(val_region)
qregmax = 0.5
qreg = np.where(qs <= qregmax)
qgamma = qs[np.min(qreg):np.max(qreg)]
print(qgamma)
print(val_start, val_end)
# qs = qs[0:50]
# print(qs)


# In[8]:


savecrossings = False
showcrossing = False
crossingstep = 2
endcrossing = 15
crossings = plot_crossings(zdata[:, :, :, 0], gs[::], qgamma, numsteps, crossingstep, endcrossing, val_start, val_end, plotdir, savecrossings, showcrossing)
# crossings = plot_crossings(gammas, gs, qgamma, numsteps, crossingstep, endcrossing, val_start, val_end, plotdir, savecrossings, showcrossing)


# In[9]:


print(crossings.shape)
print(np.where(np.isnan(crossings)))
print(qs[33])
qchoice = 34
stepc = 8
pchoice = np.argwhere(np.abs(gs - crossings[qchoice, stepc]) < 2e-3)
print(pchoice)
print(zdata[qchoice, pchoice, stepc, 0])


# In[10]:


# Plot only the consecutive crossings
close_old_plots()
plt.figure(figsize=(10,6))
plt.xlabel(r"$p_c$")
plt.ylabel(r"$q_{init}$")
plt.plot(crossings, qgamma, label=[f"RG Step {i+crossingstep} to {i+crossingstep+1}" for i in range(crossings.shape[1])[::-1]])
xu = np.array([2**((k)) for k in range(crossingstep+1, endcrossing)])
# plt.plot(xu, crossings)
print(xu)
plt.title("$q_{{init}}$ vs $p_c^{{(n)}}$")
plt.legend()
plt.show()


# In[11]:


# Fit consecutive crossings to p_c^(k) = p_c + A*2^(ky) with y = -1
fits = np.empty(shape=(qgamma.size, xu.size))
print(xu.size, crossings.shape)
r2s = np.empty(shape=(qgamma.size, xu.size), dtype=np.float64)

for a in range(endcrossing-crossingstep-2):
    for q in range(qgamma.size):
        coef = np.polyfit(xu[a:], crossings[q, a:], 1)
        y = np.polyval(coef, xu[a:])
        res = np.sum((crossings[q, a:] - y)**2)
        tot = np.sum((crossings[q, a:] - np.mean(crossings[q, a:]))**2)
        r2 = 1 - (res/tot) if tot >= 1e-12 else np.nan
        # print(r2)
        r2s[q, a] = r2
        # print(r2)
        # print(coef[-1])
        fits[q, a] = coef[-1]

# print(y, coef)
# print(fits[-1, :])
# print(np.isnan(fits).sum())


# In[12]:


print(np.isnan(r2s).sum())
print(np.where(np.isnan(r2s)))
# print(r2s[:])
print(np.isnan(fits).sum())
# print(fits[1, 5])
# print(fits.shape)
r2choice = r2s[:, :-1]
# print(r2choice[0, :])

print(np.argmax(r2choice, axis=0), np.argmax(r2choice[:, :-1], axis=1)) # For each q, which step has the best R2. For each step, which q has the best r2.
# print(r2choice[:, 1])
print(r2choice[:, 8])
print(fits[:, 9])
# print(r2s[1, :])
# print(r2s[29, :])
# for i in range(r2s.shape[1]):
#     plt.plot(qgamma, r2s[:, i], label=f"{i}")
# plt.legend()
# plt.show()


# In[13]:


# Plot the fit; the assumption that y = -1
# for i in range(endcrossing-crossingstep-2)[::-1]:
#     # print(f"i = {i}")
#     plt.plot(fits[:, i], qgamma, label=f"Fit step {i+2} to {endcrossing-1}")
plt.figure(figsize=(10,8))
plt.plot(fits[:, -3], qgamma, color="g")
plt.xlabel(r"$p_c$", fontsize=20)
plt.ylabel(r"$q_{\mathrm{init}}$", fontsize=20)
plt.yticks(np.linspace(0.0, 0.5, 11), fontsize=16)
plt.xticks(np.linspace(0.5, 0.9, 9), fontsize=16)
plt.scatter(0.5, 0.0, marker="*", color="r", s=100, label="IQHE critical point")
# plt.title(r"$q_{init}$ vs fit intercept")
plt.legend(loc="upper left", fontsize=20)
# plt.savefig(f"{plotdir}/Nu/fitted_pc.png", dpi=150)
plt.minorticks_on()
plt.grid(alpha=0.5)
# plt.savefig("./report/fitted_pc.pdf", dpi=150, bbox_inches="tight")
plt.show()
print(fits[:, -2])
# print(fits[:, 9])
# qgamma = qs[0:25]


# In[14]:


# Get the slopes for all q and n
slopes = np.empty(shape=(qgamma.size, endcrossing-crossingstep-1))
gammar2s = np.empty_like(slopes)
gammaerrs = np.empty_like(slopes)
pcs = fits[:, :-1]
pc = np.mean(pcs, axis=1)
sigmapc = np.std(pcs, axis=1)
for q in range(qgamma.size):
    for k in range(endcrossing-crossingstep-1):
        slopes[q, k], gammar2s[q, k], gammaerrs[q, k]= gamma_slope(gs[::], gammas[q, :, k], pc[q], 10, 2)
        # slopes[q, k], gammar2s[q, k] = gamma_slope(gs[::], zdata[q, :, k, 0], pc[q], 10, 2)
# for q in range(qgamma.size):
#     for k in range(endcrossing-crossingstep-1):
#         # slopes[q, k] = gamma_slope(gs, zdata[q, :, k, 0], crossings[q, k], 10, 2)
#         # slopes[q, k] = gamma_slope(gs[::], gammas[q, :, k], crossings[q, k], 10, 2)
#         slopes[q, k], gammar2s[q, k] = gamma_slope(gs[::], gammas[q, :, k], crossings[q, k], 4, 2)
#         # slopes[q, k], gammar2s[q, k] = gamma_slope(gs[::], zdata[q, :, k, 0], crossings[q, k], 10, 2)
# print(crossings[:, -1]/(1-qgamma))
# plt.plot(qgamma, crossings[:, -1]/(1-qgamma))
# plt.show()


# In[18]:


actual_k = np.array(range(crossingstep+1, slopes.shape[1]+crossingstep+1))
sys_size = np.array([2**(i) for i in actual_k])
print(f"System sizes used are k = {crossingstep+1} to {slopes.shape[1]+crossingstep}")
plt.figure(figsize=(10,8))
# Set T_k(q) = dGamma/dp at p = p_c for all q at a particular step k.
startind = 0
endind=12
thesemarkers = [",", "o", "v", "^", "s", "p", "P", "*", "+"]
for q in range(0,qgamma.size, 9):
    # plt.plot(actual_k, np.log(slopes[q, :]), label=f"q = {qgamma[q]:.3f}")
    # plt.plot(np.log(sys_size), np.log(np.abs(slopes[q, :])), label=f"q = {qgamma[q]:.3f}")
    checkslope = np.polyfit(np.log(sys_size)[startind:endind], np.log(np.abs(slopes[q, startind:endind])), 1)
    y = np.polyval(checkslope[:], logsize[startind:endind])
    ymean = np.mean(np.log(np.abs(slopes[q, startind:endind])))
    res = np.sum((np.log(np.abs(slopes[q, startind:endind])) - y)**2)
    # stderr = np.sqrt(np.sum(res**2) / (y.size-2))
    logerr = gammaerrs[q, :] / np.abs(slopes[q, :])
    # print(np.where(y))
    tot = np.sum((np.log(np.abs(slopes[q, startind:endind])) - ymean)**2)
    r = 1 - (res/tot)
    # print(r)
    # print(gammar2s[q, :])
    a = plt.plot(np.log(sys_size)[startind:endind], y, linestyle="--", alpha=0.5)
    c = a[0].get_color()
    plt.scatter(np.log(sys_size), np.log(np.abs(slopes[q,:])), s=100, label=f"q = {qgamma[q]:.3f}", marker=thesemarkers[q//7], facecolor=c, edgecolors=c)
    plt.errorbar(np.log(sys_size), np.log(np.abs(slopes[q,:])), logerr, marker="none", color=c, linestyle="none", capsize=10, alpha=0.8)
    # plt.plot(np.log(sys_size), np.log(np.abs(slopes[q, :])), label=f"q = {qgamma[q]:.3f}")
# plt.xticks(actual_k)
plt.xlabel(r"$\ln(2^k)$", fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
# plt.xticks(actual_k)
# plt.xscale("log")
plt.ylabel(r"$\ln(|T_q(k)|)$", fontsize=20)
# plt.title(r"$\ln(|T_q(k)|)$ vs RG step k")
# plt.legend(loc = "upper right", bbox_to_anchor=(1.3, 1.0))
plt.legend(loc="upper right", fontsize=14)
# plt.savefig(f"{plotdir}/Nu/slopes_vs_steps.png", dpi=150, bbox_inches="tight")
plt.minorticks_on()
plt.grid(alpha=0.5)
# plt.savefig("./report/logfit.pdf", dpi=150, bbox_inches="tight")
plt.show()


# In[16]:


# for q in range(0, qgamma.size, 2):
for q in ([0] + list(range(19, qgamma.size)))[::2]:
    plt.plot(actual_k, gammar2s[q, :], label=f"{qgamma[q]:.3f}")
plt.xticks(actual_k)
plt.xlabel("Step")
plt.ylabel(r"$R^2$")
plt.legend(loc="upper right", bbox_to_anchor = (1.3, 1.0))
plt.grid(alpha=0.4)
plt.title(r"Quality of Fit for $\frac{d\Gamma}{dp}_{|p=p_c}$ against RG step for various $q_{init}$")
# Conclusion - steps 2 or 3 must be the start, steps 7-9 should be the end.
# This is a check for the gradient at p = p_c for a step number, i.e at a certain system size.
plt.show()


# In[16]:


print(qgamma[29])


# In[53]:


# I have the slopes of dGamma/dp at p = p_c.
# logslope = np.log(slopes)
startq = 30
print(qgamma[startq])
gammaq = qgamma[startq:]
logslope = np.log(np.abs(slopes[:, :]))
logsize = np.log(sys_size)
nutests = np.empty(qgamma.size)
fitr2s = np.empty(shape=nutests.shape)
# print(logslope.shape, logsize.shape)
plt.figure(figsize = (10, 8))
endjump = 4
maxstart = endcrossing - crossingstep - endjump
ends = endcrossing - endjump
meannus = np.empty(shape=(maxstart, ends,qgamma.size-2))
stdnus = np.empty(shape = meannus.shape)
additional_ticks = []
nutesterrs = np.empty_like(nutests)
fitslopes = np.empty_like(meannus)
q0errs = np.empty(shape=(maxstart, (endcrossing-endjump), 1))
secondax = q0errs.shape[1]
k = 0
print(f"Using {qgamma[startq]:.3f}")
for startstep in range(maxstart):
    for endstep in range(startstep+endjump, endcrossing):
        for i in range(qgamma.size)[::]:
            # plt.plot(logsize, logslope[i, :])
            f = np.polyfit(logsize[startstep:endstep], logslope[i, startstep:endstep], 1)
            preds = np.polyval(f, logsize[startstep:endstep])
            ymean = np.mean(logslope[i, startstep:endstep])
            res = np.sum((logslope[i, startstep:endstep] - preds)**2)
            stderr = np.sqrt(np.sum(res**2) / (preds.size-2))
            tot = np.sum((logslope[i, startstep:endstep] - ymean)**2)
            r = 1 - (res/tot)
            fitr2s[i] = r
            slope = f[0]
            if i >= 1 and i < qgamma.size-1:
                fitslopes[startstep, endstep-endjump, i-1] = slope
            intercept = f[1]
            nutest = 1/(slope+1)
            # nutest = 1/slope
            nutests[i] = nutest
            nutesterrs[i] = stderr / ((slope+1)**2)
            # nutesterrs[i] = stderr / (slope**2)
            if i == 0:
                q0errs[startstep, endstep-startstep-endjump, 0] = stderr
            # print(nutest)
        # a = plt.scatter(qgamma[::3], nutests[::3], marker="x", label=f"Steps {startstep}-{endstep}")
        mednu = np.median(nutests[1:])
        numean = np.mean(nutests[startq:])
        leftindex = np.argwhere(nutests[:] == np.sort(nutests[:])[nutests[:].size//2 - 1]).ravel()[0]
        rightindex = np.argwhere(nutests[:] == np.sort(nutests[:])[nutests[:].size//2]).ravel()[0]
        # mednu = np.mean(nutests[:])
        # color = a.get_facecolor()[0]
        stdnu = np.std(nutests[1:])
        clippedstd = np.std(nutests[startq:])
        for i in range(1, qgamma.size)[:-1]:
            numu = np.mean(nutests[i:])
            nusig = np.std(nutests[i:])
            meannus[startstep, endstep-endjump, i-1] = numu
            stdnus[startstep, endstep-endjump, i-1] = nusig
        # if startstep in (0, 1, 2, 3, 4, 5, 6) and endstep in (3, 4, 5, 6, 7, 8, 9, 10, 11):
        #     print(f"Steps {startstep+actual_k[0]:2d} to {endstep+actual_k[0]:2d}: Med = {mednu:5.3f}, IQR[25-75] = {iqr(nutests[1:]):5.3f}, Mean = {np.mean(nutests[30:]):5.3f}, Std = {stdnu:5.3f}, IQHE = {nutests[0]:5.3f}")
        # if np.abs(numean - 2.73) <= 0.05 or np.abs(nutests[0] - 2.5) < 0.1:
        if (startstep, endstep) in [(1,8), (2, 8), (3,7), (3, 8), (1,9), (2, 9), (3,9)]:
        # if startstep in [1, 2, 3] and endstep in [6, 7, 8, 9]:
            # a = plt.errorbar(qgamma[::1], nutests[::1], clippedstd, label=f"Steps {startstep+actual_k[0]}-{endstep+actual_k[0]}, $\\nu_{{med}}$ = {np.median(nutests[30:]):.3f}, $\\bar{{\\nu}}={np.mean(nutests[30:]):.3f}$", linestyle="none", capsize=10, marker="o")
            a = plt.errorbar(qgamma[::1], nutests[::1], nutesterrs[::1], label=f"Steps ${startstep+actual_k[0]}-{endstep+actual_k[0]}$", markersize=5, linestyle="none", capsize=10, marker="o")
            color = a[0].get_color()
            additional_ticks.append(np.round(mednu, 3))

            print(f"Steps {startstep+3}-{endstep+3}: {numean:.3f} ± {clippedstd:.3f}, {(nutests[startq]+nutests[startq+1])/2:.3f} ± {(nutesterrs[startq]+nutesterrs[startq+1])/2:.3f}, {nutests[0]:.3f} ± {nutesterrs[0]:.3f}")
            print(f"Nu at q = {qgamma[startq]:.3f}: {nutests[startq]:.3f} ± {nutesterrs[startq]:.3f}")
            if (startstep, endstep) == (2, 8):
                chosen = np.mean(nutests[startq:])
                chosenlabel = f"$\\bar{{\\nu}}_{{q\\geq q_{{\\mathrm{{cut}}}}}}={np.mean(nutests[startq:]):.3f} \\pm {clippedstd:.3f}$"
                c = color
            # e = nutesterrs[0]
plt.axhline(2.73, linestyle="--", color="g")
plt.axhspan(2.73-0.02, 2.73+0.02, alpha=.3, color="g", linestyle="--", label=r"$2.73 \pm 0.02$")
    # print(nutests[0])
plt.axvspan(0.1, 0.3,2.5, 2.9, alpha=0.4, linestyle="-.")
plt.axhline(chosen, linestyle="--", color=c, label=chosenlabel)
plt.axvline(qgamma[30], linestyle="--", color="k", label=r"$q_{\mathrm{cut}}=0.306$", alpha=0.5)
plt.xlabel(r"$q_{\mathrm{init}}$", fontsize=20)
plt.ylabel(r"$\nu$", fontsize=20)
# plt.title("$\\nu$ vs $q_{{init}}$ for the QSHE")
# plt.xlim(qgamma[startq-1], 0.5)
# c = plt.gca().get_yticks()
# newticks = sorted(list(c) + additional_ticks)
# plt.yticks(ticks=newticks)
# print(sys_size)
plt.xticks([0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50], fontsize=16)
plt.yticks(sorted(list(np.linspace(2.2, 3.6, 10)) + [2.73]), fontsize=16)
plt.ylim((2.15, 3.65))
plt.minorticks_on()
# plt.legend(bbox_to_anchor=(1.4, 1.0))
plt.legend(loc="upper right", fontsize=14, bbox_to_anchor=(1.4, 1.0))
# plt.tight_layout()
plt.grid(alpha=0.5)
# plt.savefig("./report/nu_qshe.pdf", bbox_inches="tight")
plt.show()
close_old_plots()


# In[222]:


# print(meannus.shape)
plt.figure(figsize=(10, 8))
print(startstep, endstep)
origshape = meannus.shape
print(origshape)
valid = ~np.isnan(meannus) & ~np.isnan(stdnus)
valid &= (meannus > 2.4) & (meannus < 2.8)
valid &= (stdnus < 0.05)
newnus = np.where(valid, meannus, np.nan)
newstds = np.where(valid, stdnus, np.nan)
# If a is 5, b goes through the range [5+5, 14]. The actual range in k is then [a+3, b+3]. So if [a, b] = [5, 8], then a_k, b_k = [5+3, ]
goodwindows = []
# print(markers)
for a in range(origshape[0])[::]:
    marker = markers[::][a%len(markers)]
    for b in range(origshape[1])[::]:
        frac = np.mean(valid[a, b, :])
        c = a + endjump + b + actual_k[0]
        if frac > 0.3:
            if a == 1:
                continue
            # print(f"start={a+3}, end={b+a+3}: {frac:.2f} valid")
            goodwindows.append((a, b))
            print(newnus[a,b,30])
            plt.errorbar(qgamma[1:-1], newnus[a, b, :], np.abs(newstds[a, b, :]), linestyle="none", capsize=10, markersize=8, marker=marker, label=f"Steps {actual_k[a]}-{actual_k[b+endjump]}")
# stdnus = np.clip(stdnus, )
# print(np.argwhere(stdnus < 0))
# print(stdnus[1, 0, 0], meannus[1, 0, 0])
# plt.errorbar(qgamma[1:-1], meannus[a, b, :], np.abs(stdnus[a, b, :]), linestyle="none", capsize=10, marker="o")
plt.xlabel(r"$q_{\mathrm{cut}}$", fontsize=20)
plt.ylabel(r"$\bar{\nu}$", fontsize=20)
plt.axhline(2.73, linestyle="--", color="g", alpha=0.6,zorder=2)
plt.axhspan(2.73-0.02, 2.73+0.02, color="g", linestyle="--", alpha=0.2, label=r"$\nu = 2.73\pm0.02$", zorder=2)
plt.axvline(qgamma[30], linestyle="--", color="k", alpha=0.6, label=r"$q_{\mathrm{cut}} = 0.306$", zorder=2)
# plt.yticks(list(np.linspace(2.68, 2.78, 11)), fontsize=16)
plt.legend(bbox_to_anchor=(1.0, 1.0), fontsize=14)
plt.xticks(np.round(np.linspace(0.2, 0.48, 15), 2), fontsize=16)
plt.yticks(list(np.round(np.linspace(2.4, 3.0, 7),2))+[2.73], [str(y) for y in list(np.round(np.linspace(2.4, 3.0, 7),2))+[2.73]], fontsize=16)
plt.grid(alpha=0.5)
plt.ylim((2.4, 3.0))
plt.xlim((0.19, 0.49))
plt.minorticks_on()
plt.savefig("./report/qcut.pdf",dpi=150, bbox_inches="tight")
plt.show()
close_old_plots()


# In[507]:


qcutsum = []
for s, e in goodwindows:
    mu = newnus[s, e, :]  # shape (47,)
    # finite difference of the mean
    dmu = np.diff(mu)
    # find first index where slope is small
    norm_gradient = np.abs(dmu) / newstds[s, e, :-1]
    # flat_idx = np.where(np.abs(dmu) < newstds[s, e, ])[0]
    # flat_idx = np.argmax(norm_gradient < 1.0)
    flat_idx = np.argmax(np.abs(dmu) < 0.08*newstds[s, e, :-1])
    print(flat_idx)
    # if len(flat_idx) > 0:
    q_cut_auto = qgamma[1:-1][flat_idx]
    qcutsum.append(q_cut_auto)
    print(f"Steps {s+3}-{e+3+endjump}: q_cut ≈ {q_cut_auto:.3f}, "
            f"ν = {mu[flat_idx]:.3f} ± {newstds[s, e, flat_idx]:.3f}")
print(np.mean(qcutsum))
print(np.argwhere(np.abs(qgamma-np.mean(qcutsum))<5e-3))
print(qgamma[29], qgamma[30])


# In[27]:


slopediffs = logslope[:, 1:] - logslope[:, :-1]
nueff = 1 / ((slopediffs/np.log(2))+1)
# print(logsize)
# slopediffs = logslope[0, :] / logsize
testslope = np.polyfit(logsize[2:8], logslope[0, 2:8], 1)[-2]
testnu = 1 / (testslope+1)
print(testnu)
# print(1 / (testslope+1))
# print(nueff.shape)


# In[221]:


# print(q0errs.shape)
# print(np.argwhere(q0errs == e))
# print(q0errs[2, 5])
plt.figure(figsize=(8, 6))
for i in range(q0errs.shape[0])[::2]:
    plt.scatter(actual_k, q0errs[i, :, 0], label=f"Start = {i}", marker="x")
plt.ylim((-0.1, 1.0))
plt.xticks(actual_k)
plt.xlabel("End step")
plt.ylabel("Regression error")
plt.title("Regression error of linear fit for q = 0")
# plt.legend(bbox_to_anchor=(1.1, 1.0))
plt.legend(loc="upper left")
plt.show()


# In[30]:


# print(len(loaded_colors))
# colors = list(loaded_colors.values())
print(nueff.shape)


# In[51]:


# qc = 0
# print(nueff[qc, :])
plt.figure(figsize=(10, 8))
qstep = 1
if len(markers) < qgamma.size // qstep:
    markers = markers + markers
goodchoices = np.argwhere(np.abs(nueff - 2.73) <= 0.04)
# print(goodchoices.shape)
# print(np.unique(goodchoices[:, 1], return_counts=True))
# print(goodchoices)
# print(qgamma[goodchoices[:, 0]])
# print(actual_k[goodchoices[:, 1]])
vals = np.array([qgamma[goodchoices[:, 0]], actual_k[goodchoices[:, 1]]])
# print(np.round(vals, 3))
nueffs = nueff[1:, :]
# print(np.median(nueff, axis=0), np.median(nueffs, axis=0))
# print(np.mean(nueff, axis=0), np.mean(nueffs, axis=0))
mediannu = np.median(nueffs, axis=0)
meannu = np.mean(nueffs, axis=0)
# print(iqr(nueffs, axis=0))
stdnu = np.std(nueffs, axis=0)
# print(stdnu)
for qc in range(qgamma.size)[1::qstep]:
    # plt.scatter(actual_k[:-1], nueff[qc, :], marker = "+")
    # plt.scatter(actual_k[:-1], nueff[qc, :], label=f"q = {qgamma[qc]:.3f}", marker=markers[qc//qstep], s=50)
    plt.scatter(actual_k[:-1], nueff[qc, :], marker="o", facecolors="none", edgecolors= "b", s=100, zorder=1)
plt.scatter(actual_k[:-1], nueff[0, :], marker="*", color="r", s=150, label="q = 0.0", zorder=2)
seeq = 30
plt.scatter(actual_k[:-1], nueff[seeq, :], marker="s", color="g", s=250, label=f"q = {qgamma[seeq]:.3f}", zorder=3, alpha=0.6)
# print(nueff[0, 4])
print((nueff[30]+nueff[31])/2, qgamma[32])
# plt.errorbar(actual_k[:-1], meannu, stdnu, linestyle="none", marker="o", color="b", capsize=10, label="QSHE Means", zorder=1)
# plt.errorbar(actual_k[:-1], mediannu, iqr(nueffs, axis=0), linestyle="none", marker="^", color="g", capsize=10, label="QSHE Medians", zorder=2, markersize=10)
# plt.scatter(actual_k[:-1], nueff[0, :], marker="*", color="r", label="IQHE", s=100, zorder=3)
# plt.scatter(actual_k[:-1], mediannu, marker="+", color="g", label="QSHE Medians", s=100)
# plt.scatter(actual_k[:-1], meannu, marker="*", color="r", label="Means")
# print(mediannu)
# print(nueff[goodchoices])
indexes = goodchoices[:, 0]
ks = goodchoices[:, 1]
# plt.scatter(actual_k[0]+ks, nueff[indexes, ks], marker="o", color="k", s=200, alpha=0.5)
maxy = 7
plt.ylim((0.95, maxy+0.05))
plt.ylabel(r"$\nu_{\mathrm{eff}}$", fontsize=20)
plt.xlabel(r"RG step $(k)$", fontsize=20)
plt.xticks(actual_k[:-1], fontsize=16)
plt.yticks(list(np.linspace(1.0, maxy, 7))+[2.51, 2.73], fontsize=16)
plt.minorticks_on()
plt.axhline(2.73, linestyle="--", color="m", label="QSHE 2.73", zorder=0)
plt.axhline(2.51, linestyle="--", color="g", label="IQHE 2.51", zorder=0)
plt.legend(loc="lower right", fontsize=16)
# plt.legend(bbox_to_anchor=(1.2, 1.0))
plt.tight_layout()
# plt.savefig("./report/nu_eff_allq.pdf")
plt.show()
close_old_plots()


# In[296]:


plt.figure(figsize=(8, 6))
qsee = 23
pcs = fits[:, :-1]
print(fits[qsee, :])
pc = np.mean(pcs, axis=1)
sigmapc = np.std(pcs, axis=1)
# print(pc, sigmapc)
# plt.errorbar(qgamma, pc, sigmapc, linestyle="none", marker="o", color="r", capsize=10)
for k in actual_k[:-1]:
    plt.plot(gs, zdata[qsee, :, k, 0], label=f"Step {k}")
    x = pc * (sys_size[k - actual_k[0]] ** (1 /np.mean(nueff, axis=0)[k-actual_k[0]]))


# print(np.argwhere(np.abs(gs - pc[qsee]) <= 2e-3))
# print(pc[qsee])
plt.axvline(pc[qsee], linestyle="--", color="r", alpha=0.5, label=f"$p_c$ = {pc[qsee]:.3f}")
# plt.xlim((pc[qsee] - 0.001, pc[qsee]+0.001))
# plt.ylim((-2.0, 0.0))
plt.xlabel("p")
# plt.xticks(qgamma[::5])
plt.ylabel("z")
plt.title(f"z(p) vs p for q = {qgamma[qsee]:.3f}")
plt.tight_layout()
plt.grid(alpha=0.5)
plt.legend(bbox_to_anchor=(1.25, 1.0))
plt.show()
close_old_plots()


# In[117]:


print(fitr2s)


# ## Deprecated

# In[172]:


endstep = 7
qview = 0
for e in range(endstep, endstep+5):
    for s in range(endstep-3):
        # f = np.polyfit(logsize[s:endstep], logslope[qview, s:endstep], 1)
        f = np.polyfit(logsize[s:endstep], logslope[qview, s:endstep], 1)
        slope = f[0]
        intercept = f[1]
        nutest = 1/(slope+1)
        preds = np.polyval(f, logsize[s:endstep])
        ymean = np.mean(logslope[qview, s:endstep])
        res = np.sum((logslope[qview, s:endstep] - preds)**2)
        tot = np.sum((logslope[qview, s:endstep] - ymean)**2)
        r = 1 - (res/tot)
        nutests[i] = nutest
        plt.scatter(s, nutest)
        # plt.scatter(s, r, marker="+", label=f"{s+actual_k[0]}-{endstep+actual_k[0]}")
        # print(r)
        # print(f"R2 = {r}, Nu = {nutest}, start = {s+actual_k[0]}, end = {endstep+actual_k[0]}")
        print(nutest, s, e)

plt.title(f"{qgamma[qview]:.3f}")
plt.show()


# In[126]:


nus = np.empty(shape=(qgamma.size, endcrossing-crossingstep-1))
nuslopes = np.empty(shape=(qgamma.size, endcrossing-crossingstep-1))
nuspread = np.empty(shape=(qgamma.size))
for stepchoice in range(crossingstep+1, endcrossing):
    for q in range(qgamma.size):
        # nu, m, b = fit_nu(actual_k, sys_size*slopes[q, :], stepchoice)
        nu, m, b = fit_nu(actual_k, slopes[q, :], stepchoice)
        # nus[q, stepchoice-crossingstep-1] = nu
        nus[q, stepchoice-crossingstep-1] = nu
        nuslopes[q, stepchoice-crossingstep-1] = m
nuspread = np.abs(nus[:, -2] - nus[:, 0])
# print(nuspread)


# In[127]:


print(nus.shape)
print(nus[::3, :])
# print(*range(crossingstep, endcrossing-1))


# ### Disorder potential

# In[ ]:





# ### Nu

# In[124]:


viewchoice = 0
qstep = 5
for i in range(nus.shape[1]):
    # plt.plot(qs, nus[:, i], label=f"Steps {i+3} to {numsteps-1}")
    plt.errorbar(qgamma[::qstep], nus[::qstep, i], yerr=nuspread[::qstep], linestyle="none", marker="o", label=f"Steps {i+crossingstep} to {endcrossing-1}", capsize=10)
    print(f"Nu for q = {qgamma[viewchoice]:.3f} at step {i+3} = {nus[viewchoice, i]}, err = {nuspread[i]}, {nuspread[i]*100/nus[viewchoice, i]:.2f}%")
plt.legend()
# plt.xlim((-0.02, 0.42))
plt.xlabel(r"$q_{init}$")
plt.ylabel(r"$\nu$")
plt.xticks(np.round(qgamma[::qstep], 2))
plt.yticks(range(-1, 10))
plt.ylim((-1, 10))
plt.grid(alpha=0.5)
plt.title(r"$\nu$ vs $q_{init}$")
plt.savefig(f"{plotdir}/Nu/nu_vs_q.png", dpi=150)
plt.show()


# In[351]:


close_old_plots()
plt.errorbar(sys_size[:-1], nus[viewchoice, :-1], yerr = nuspread[viewchoice], marker="o", color="r", capsize=5, linestyle="none", label=f"QSHE, q={qs[viewchoice]:.3f}")
plt.axhline(2.51, linestyle="--", label="IQHE = 2.51")
plt.legend(loc="lower right")
plt.title(f"$\\nu$ vs system size for q = {qs[viewchoice]:.3f}")
plt.xlabel("system size")
plt.ylabel(r"$\nu$")
plt.xticks(sys_size[:-1])
plt.ylim((1.5, 3.0))
plt.savefig(f"{plotdir}/Nu/nu_vs_system_size_q{qs[viewchoice]:.3f}.png", dpi=150)
plt.show()


# In[372]:


from scipy.stats import norm
from source.fitters import fit_z_peaks
# print(zdata.shape)
steprange = range(1, numsteps + 1)
choiceq = 0
choicep = np.argwhere(np.abs(gs - fits[:, :-1][choiceq, -2]) < 2e-3).ravel()[0]
# choicep = 420
print(choicep)
choicepsep = 5
startp = choicep
endp = choicep - choicepsep
# for i in range(choicep-choicepsep, choicep+choicepsep):
#     plt.plot(steprange, zdata[choiceq, i, :, 1], label=f"{gs[i]:.3f}")
close_old_plots()
# pchoices = gs[choicep:choicep+choicepsep]
pchoices = gs[startp:endp:-1]
print(pchoices[::-1])
psep = (cfg.p_max - cfg.p_min) / cfg.p_num
gchoices = pchoices + (1 - pchoices)*qs[choiceq]
zchoices = np.log((1-gchoices)/gchoices)
zshift = (np.max(zchoices) - np.min(zchoices))/zchoices.size
# zaxis = np.abs(zchoices - zchoices[0])
zaxis = np.array(range(pchoices.size)) * zshift
print(zaxis)
# print(zchoices[0], zchoices[-1])
# print(pchoices[0], pchoices[-1])
# print(zchoices)
# print(zshift)
# print(zaxis)
# print(pchoices)
print(f"Analysis for p = {gs[choicep]}")
xs = steprange
fitmeans = np.empty((pchoices.size, len(xs)))
fitstds = np.empty((pchoices.size, len(xs)))
# print(pchoices)
# print(pchoices)
for k in range(pchoices.size):
    i = startp - k
    for m in steprange:
        j = m-steprange[0]
        x = np.linspace(zdata[choiceq, i, j, 0] - 3*zdata[choiceq, i, j, 2], zdata[choiceq, i, j, 0] + 3*zdata[choiceq, i, j, 2], 1000) # Plot the data simply over a range of 4 STDs
        thismean = zdata[choiceq, i, j, 0]
        thisstd = zdata[choiceq, i, j, 2]
        # fitmean, fitstd = norm.fit(x, loc=zdata[plotq, i, steprange[0]:steprange[-1], 0], scale=zdata[plotq, i, steprange[0]:steprange[-1], 2])
        # I want to fit a gaussian to a range of 2.5 std around the mean of z, for each RG step, at this p value
        # fitmean, fitstd = norm.fit(x, loc=thismean, scale=thisstd)
        fitmean, fitstd = norm.fit(x)
        fitmeans[k, j] = fitmean
        fitstds[k, j] = fitstd
# print(fitmeans[:, 0])
# print(fitmeans[0, :])
# print(fitmeans[:, 5] - fitmeans[0, 5])
for i in range(numsteps)[:12:1]:
    # zvals = np.abs(zdata[choiceq, startp:endp:-1, i, 0] - zdata[choiceq, startp, i, 0])
    zvals = fitmeans[:, i]
    # zvals = fitmeans[:, i] - fitmeans[0, i]
    # print(fitmeans[:, i])
    # print(fitmeans[:, i] - fitmeans[0, i])
    # zvals = zdata[choiceq, startp:endp:-1, i, 0]
    s, r2 = fit_z_peaks(zaxis[1:], zvals[1:])
    # fiterrs = fitstds[:, i] / np.sqrt(cfg.samples)
    fiterrs = fitstds[:, i] / np.sqrt(500)
    a = plt.errorbar(zaxis[1:], zvals[1:], yerr=fiterrs[1:], linestyle="none", label=f"RG {i+1}", marker="o", capsize=10)
    c = a.lines[0].get_color()
    # plt.scatter(zaxis, zvals, label=f"RG {i+1}")
    # plt.plot(zaxis, s*zaxis + fitmeans[0, i], linestyle="--", alpha=0.5, color=c)
    plt.plot(zaxis, s*zaxis, linestyle="--", alpha=0.5, color=c)
    nuval = np.log((2**(i+2)))/np.log(s)
    print(s, r2, nuval)
# plt.axhline(zdata[choiceq, choicep, 0, 1], linestyle="--")
# plt.axvline(zchoices[0], linestyle="--", color="r")
#     plt.plot(pchoices, zdata[choiceq, startp:endp:-1, i, 0], label=f"RG {i+1}")
    # plt.plot(gs, zdata[choiceq, :, i, 0], label=f"RG {i+1}")
    # plt.plot(gs, gammas[choiceq, :, i])
# plt.xlim((0.7, 0.9))
plt.title(f"q = {qs[choiceq]:.3f}")
plt.xlabel(r"$z_0$")
plt.ylabel(r"$\bar{z}$")
# plt.ylim((-5.0, 5.0))
plt.legend(loc="upper right", bbox_to_anchor=(1.2, 1.0))
plt.show()


# In[378]:


def get_nus(numsteps, fits, ps, choiceq, zdata):
    close_old_plots()
    steprange = range(0, numsteps + 1)
    # choiceq = 31
    # clippedfits = np.clip(fits[:, :-1], 1e-9, None)
    clippedfits = fits[:, :-1]
    choicep = np.argwhere(np.abs(ps - clippedfits[choiceq, -2]) < 2e-3).ravel()[0] +1
    # print(ps[choicep])
    # print(choicep)
    choicepsep = 10
    startp = choicep
    endp = choicep - choicepsep
    pchoices = ps[startp: endp:-1]
    gchoices = pchoices + (1 - pchoices)*qs[choiceq]
    zchoices = np.log((1-gchoices)/gchoices)
    zshift = (np.max(zchoices) - np.min(zchoices))/zchoices.size
    zaxis = np.abs(zchoices - zchoices[0])
    # print(zaxis)
    met = np.empty(shape=(numsteps, 4))
    xs = steprange
    fitmeans = np.empty((pchoices.size, len(xs)))
    fitstds = np.empty((pchoices.size, len(xs)))
    # print(pchoices)
    # print(pchoices)
    for k in range(pchoices.size):
        i = startp - k
        for m in steprange:
            j = m-steprange[0]
            x = np.linspace(zdata[choiceq, i, j, 0] - 4.5*zdata[choiceq, i, j, 2], zdata[choiceq, i, j, 0] + 4.5*zdata[choiceq, i, j, 2], 200) # Plot the data simply over a range of 4 STDs
            thismean = zdata[choiceq, i, j, 0]
            thisstd = zdata[choiceq, i, j, 2]
            # fitmean, fitstd = norm.fit(x, loc=zdata[plotq, i, steprange[0]:steprange[-1], 0], scale=zdata[plotq, i, steprange[0]:steprange[-1], 2])
            # I want to fit a gaussian to a range of 2.5 std around the mean of z, for each RG step, at this p value
            fitmean, fitstd = norm.fit(x, loc=thismean, scale=thisstd)
            fitmeans[k, j] = fitmean
            fitstds[k, j] = fitstd
            # if k == 0:
            #     print(f"p = {gs[i]:.3f}, q = {qgamma[choiceq]}, step = {j}")
            #     print(f"Actuals: Median = {zdata[choiceq, i, j, 0]:.3f}, Mean = {zdata[choiceq, i, j, 1]:.3f}, STD = {zdata[choiceq, i, j, 2]:.3f}")
            #     print(f"Fits: Mean = {fitmean:.3f}, STD = {fitstd:.3f}")
            #     # print(f"Max err = {np.abs(fitmeans[k, :] - zdata[choiceq, i, steprange[0]:steprange[-1]+1, 0])}")
            #     print("-"*80)
    # print(fitmeans)
    for i in range(numsteps):
        # zvals = np.abs(zdata[choiceq, startp:endp:-1, i, 0] - zdata[choiceq, startp, i, 0])
        # zvals = np.abs(zdata[choiceq, choicep:choicep+choicepsep, i, 0])
        # zvals = fitmeans[:, i]
        zvals = fitmeans[:, i] - fitmeans[0, i]
        # plt.scatter(pchoices, zvals, label=f"RG {i+1}")
        s, r2 = fit_z_peaks(zaxis[:], zvals[:])
        serr = np.sqrt(((zvals - s*zaxis)**2).sum()/(zaxis.size - 1) / (zaxis**2).sum())
        nuval = np.log((2**(i+1)))/np.log(s)
        met[i, :] = np.array([s, r2, nuval, serr])
        # if i > 2:
        #     print(f"Step {i+1}: Slope = {s}, R2 = {r2}, Nu = {nuval}")
    return met

# mets = np.empty(shape=(fits.shape[0], numsteps, 4))
endstep = 9
mets = np.empty(shape=(fits.shape[0], endstep, 4))
for i in range(qgamma.size):
    # mets[i, :, :] = get_nus(numsteps, fits, gs, i, zdata)
    mets[i, :, :] = get_nus(endstep, fits, gs, i, zdata)
# plt.axhline(zdata[choiceq, choicep, 0, 1], linestyle="--")
# plt.axvline(gs[choicep], linestyle="--", color="r")
# plt.title(f"q = {qs[choiceq]:.3f}")
# plt.xlabel("p_init")
# plt.ylabel(r"$\bar{z}$")
# plt.legend(loc="upper right", bbox_to_anchor=(1.2, 1.0))
# plt.show()


# ### TEST

# In[379]:


# print(mets.shape)
# print(np.unique(mets[:, :, 2], return_index=True))
qsee = 0
# print(mets[qsee, :, 0])
# print(mets[qsee, :, 1])
# print(mets[qsee, :, 2])
# print(mets[0, qsee, :, 0])
# print(mets[0, qsee, :, 1])
# print(mets[0, qsee, :, 2])
# os.makedirs(f"{plotdir}/Nu/Fits", exist_ok=True)
close_old_plots()
# for qsee in range(qgamma.size):
#     fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 9))
#     ax0.set_xlabel("RG step")
#     ax1.set_xlabel("RG step")
#     ax0.set_ylabel("R2")
#     ax1.set_ylabel(r"$\nu$")
#     for fn in fitnums[:]:
#         ax0.scatter(range(mets.shape[2])[1:-1], mets[fn-1, qsee, 1:-1, 1], label=f"Fitnum = {fn}")
#         ax1.scatter(range(mets.shape[2])[1:-1], mets[fn-1, qsee, 1:-1, 2], label=f"Fitnum = {fn}")
#     avgnu = np.median(mets[:, qsee, 1:-1, 2])
#     ax1.axhline(avgnu, linestyle="--", color="r", label=f"Avg = {avgnu:.3f}")
#     ax1.set_ylim((2.0, 5.0))
#     ax0.set_ylim((0.98, 1.01))
#     ax0.legend(loc="lower left")
#     ax1.legend(loc="upper right", bbox_to_anchor=(1.25, 1.0))
#     ax0.set_title(r"$R^2$")
#     ax1.set_title(r"$\nu$")
#     plt.suptitle(f"$R^2$ and $\\nu$ for q = {qgamma[qsee]:.3f}")
#     fig.savefig(f"{plotdir}/Nu/Fits/fits_{qgamma[qsee]:.3f}.png", dpi=150, bbox_inches="tight")
#     # plt.show()
#     close_old_plots()
# for qsee in range(qgamma.size):
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 9))
ax0.set_xlabel("RG step")
ax1.set_xlabel("RG step")
ax0.set_ylabel("R2")
ax1.set_ylabel(r"$\nu$")
avgnu = np.median(mets[qsee, 1:-1, 2])
ax1.axhline(avgnu, linestyle="--", color="r", label=f"Avg = {avgnu:.3f}")
for qsee in range(qgamma.size)[::3]:
    ax0.scatter(range(mets.shape[1])[1:-1], mets[qsee, 1:-1, 1], label=f"{qgamma[qsee]:.3f}")
    ax1.scatter(range(mets.shape[1])[1:-1], mets[qsee, 1:-1, 2], label=f"{qgamma[qsee]:.3f}")
ax1.set_ylim((2.0, 5.0))
ax0.set_ylim((0.98, 1.00))
ax0.legend(loc="lower left")
ax1.legend(loc="upper right", bbox_to_anchor=(1.35, 1.0))
ax0.set_title(r"$R^2$")
ax1.set_title(r"$\nu$")
plt.suptitle(f"$R^2$ and $\\nu$ for q = {qgamma[qsee]:.3f}")
# fig.savefig(f"{plotdir}/Nu/Fits/fits_{qgamma[qsee]:.3f}.png", dpi=150, bbox_inches="tight")
plt.show()
close_old_plots()


# ### Back to normal

# In[380]:


close_old_plots()
print(mets[0, :, 2])
print(mets[0, :, 0])
numets = mets[:, :, 2]
serrors = mets[:, :, 3]
numeterrs = np.empty(shape=numets.shape)
# print(numets.shape)
for i in range(endstep):
    err = serrors[:, i] * np.log(2**i)/(mets[:, i, 0] * (np.log(mets[:, i, 0])**2))
    # print(err)
    numeterrs[:, i] = err
rmets = mets[:, :, 1]
# numeterrs = np.max(numets, axis=1) - np.min(numets, axis=1)
# rmeterrs = np.max(rmets, axis=1) - np.min(rmets, axis=1)
# numeterrs = np.std(numets, axis=1) / np.sqrt(cfg.samples)

rmeterrs = np.std(rmets, axis=1)
# print(numeterrs.shape)
# print(numets[:, 11])
# print(numeterrs)
# print(np.min(numets[12]), np.max(numets[12]))
# plt.scatter(range(numsteps), mets[25, :, 2])
# plt.show()


# In[383]:


# Nus
# plt.errorbar(qgamma, np.unique(numets[:, 0]), numeterrs,linestyle="none", capsize=20)
qview = 32
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 6))
h0 = ax0.axhline(2.513, linestyle="--", color="r", label="Shaw & Roemer, 2.513", alpha=0.5)
h1 = ax0.axhline(2.593, linestyle="--", color="m", label="Slevin & Ohtsuki, 2.593", alpha=0.5)
ax0.set_title(f"Nu vs RG steps for q = {qgamma[qview]:.3f}")
ax0.set_xlabel("RG steps")
ax0.set_ylabel("Nu")
ax0.errorbar(range(1, endstep), numets[qview, 1:], numeterrs[qview, 1:], marker="o", linestyle="none", capsize=10, color="b")
ax1.set_title(f"R2 vs RG steps for q = {qgamma[qview]:.3f}")
ax1.set_xlabel("RG steps")
ax1.set_ylabel("R2")
ax1.scatter(range(1, endstep), rmets[qview, 1:], marker="o", color="r")
print(numets[qview])
print(rmets[qview])
print(numeterrs[qview])
ax0.legend()
plt.show()
# plt.show()


# In[385]:


# close_old_plots()
# nustep = 10
indexskip = 3
fig, ax = plt.subplots(figsize=(9, 6))
startnu = 3
endnu = 9
for nustep in range(startnu, endnu):
    plt.errorbar(qgamma[::indexskip], numets[::indexskip, nustep], numeterrs[::indexskip, nustep], linestyle="none", capsize=15, marker="o", label=f"RG step {nustep}")
# plt.errorbar(qgamma[::indexskip], numets[::indexskip, nustep], numeterrs[::indexskip], linestyle="none", capsize=15, marker="o", label=f"RG step {nustep}")
# plt.yticks(np.round(np.linspace(np.min(numets[:, nustep]), np.max(numets[:, nustep]), 10), 3))
plt.xticks(np.round(np.linspace(np.min(qgamma), np.max(qgamma), 10), 3))
plt.xlabel(r"$q_{init}$")
plt.ylabel(r"$\nu$")
plt.grid(alpha=0.5)
plt.axhline(2.73, linestyle="--", color="m", alpha=0.5, label=r"Kobayashi $\it{et\; al.}$")
# plt.axhline()
gap = 2.73 - np.min(numets[::indexskip, 5:]) * 1.01
up20 = 2.73 * 1.2
up10 = 2.73 * 1.1
plt.axhline(up10, linestyle="--", color="g", alpha=0.5)
plt.axhline(up20, linestyle="--", color="g", alpha=0.5)
plt.axhline(2.73*1.3, linestyle="--", color="g", alpha=0.5)
plt.axhline(2.73*1.4, linestyle="--", color="g", alpha=0.5)
# ax.set_yticks(np.linspace(2.73-gap, np.max(numets[::indexskip, 6:])*1.01, 8))
# cticks = ax.get_yticks()
# newticks = sorted(list(np.round(cticks, 3))+[2.73])
plt.ylim((2.0, 4.0))
# plt.yticks(ticks=newticks)
plt.legend(loc="lower right")
plt.title(f"$\\nu$ vs $q_{{\\text{{init}}}}$")
# fig.savefig(f"{root_dir}/report/nu_vs_q.pdf")
plt.show()


# In[289]:


# Nus
# plt.errorbar(qgamma, np.unique(numets[:, 0]), numeterrs,linestyle="none", capsize=20)
qview = 0
fig, ax0= plt.subplots(figsize=(9, 6))
ax0.axhline(2.513, linestyle="--", color="r", label="Shaw & Roemer", alpha=0.5)
ax0.axhline(2.593, linestyle="--", color="m", label="Slevin & Ohtsuki", alpha=0.5)
ax0.set_title(f"Nu vs RG steps for q = {qgamma[qview]:.3f}")
ax0.set_xlabel("RG steps")
ax0.set_ylabel("Nu")
ax0.errorbar(range(1, numsteps-1), numets[qview, 1:-1], 2*numeterrs[qview], marker="o", linestyle="none", capsize=10, color="b")
ax0.errorbar(numsteps-1, numets[qview, -1], 2*numeterrs[qview], linestyle="none", capsize=10, color="g")
ax0.scatter(numsteps-1, numets[qview, -1], marker="*", color="g", s=90, label=f"This work, {numets[qview, -1]:.3f} $\\pm {numeterrs[qview]:.3f}$")
print(numets[qview])
print(rmets[qview])
print(numeterrs[-1])
labels = ["Shaw & Roemer, 2.513", "Slevin & Ohtsuki", f"This work, {numets[qview, -1]:.3f} $\\pm {numeterrs[qview]:.3f}$"]
ax0.legend(labels=labels)
ticks = [2.513, 2.593]
t = np.concat([np.array(ticks), np.linspace(2.673, np.max(numets[qview, 1:]*1.01), 10)])
# ax0.set_yticks(ticks=np.linspace(np.min(numets[qview, 1:])*0.99, np.max(numets[qview, 1:]*1.01), 12))
ax0.set_yticks(t)
# plt.show()
fig.savefig(f"{root_dir}/report/qshe_nu_q0.pdf")
# plt.show()


# In[949]:


def S_node(t, r, f):
    S = np.zeros(shape=(t.size, 4, 4), dtype=np.complex128)
    S[:, 0, 0] = t
    S[:, 0, 1] = r
    S[:, 0, 3] = f
    S[:, 1, 1] = -t
    S[:, 1, 0] = r
    S[:, 1, 2] = -f
    S[:, 2, 2] = t
    S[:, 2, 3] = r
    S[:, 2, 1] = -f
    S[:, 3, 3] = -t
    S[:, 3, 2] = r
    S[:, 3, 0] = f

    S_transpose = np.conjugate(np.transpose(S, (0, 2, 1)))
    id_matrix = np.eye(4, dtype=np.complex128)[None, :, :]

    projection = S_transpose @ S
    error = np.max(np.abs(projection - id_matrix))
    print(f"Max error = {error}")
    assert np.allclose(id_matrix, projection, atol=1e-12, rtol=0)
    print("The S matrix is unitary")

inp = np.array([1.0, 0.0, 0.0, 0.0])
N = 1000
rng = build_rng(1234)
indices = rng.integers(0, N, (N, 5))
tstart = rng.uniform(0, 1, N)
fstart = np.sqrt((1 - tstart**2)*0.2)
ts = np.take(tstart, indices)
fs = np.take(fstart, indices)
# ts = generate_constant_array(10000, 0.5, 5)
# fs = generate_constant_array(10000, 0.5, 5)
rs = np.sqrt(1 - ts**2 - fs**2)
# phis = generate_constant_array(100, np.pi*1.2, 16)
phis = generate_random_phases(N, rng, 16)
d = solve_qshe_matrix(ts, fs, phis, N, [2, 9, 10, 17], inp)
# d[2] = np.clip(d[2], 1e-12, 1-1e-12)
# d[9] = np.clip(d[9], 1e-12, 1-1e-12)
# d[10] = np.clip(d[10], 1e-12, 1-1e-12)
# d[17] = np.clip(d[17], 1e-12, 1-1e-12)
# t2 = np.take(d[9], indices)
# f2 = np.take(d[17], indices)
# d = solve_qshe_matrix(t2, f2, phis, N, [2,9,10,17], [1.0, 0.0, 0.0, 0.0])


# In[950]:


S_node(ts[:,0], rs[:,0], fs[:,0])
print(np.abs(d[10]).mean())
taup = np.sqrt(np.abs(d[9]) **2 + np.abs(d[10])**2).ravel()
m = np.sqrt(1 - np.abs(d[2])**2 - np.abs(d[17])**2).ravel()
# print(taup, m)
# S_node(np.abs(d[2]).ravel(), np.abs(d[9]).ravel(), np.abs(d[17]).ravel())
S_node(np.abs(d[2]).ravel(), np.abs(taup), np.abs(d[17]).ravel())
S_node(np.abs(d[2]).ravel(), m, np.abs(d[17]).ravel())


# In[971]:


def single_node(inputs, t, r, f):
    n = t.size
    S = np.zeros(shape=(n, 4, 4), dtype=np.complex128)
    S[:, 0, 0] = t
    S[:, 0, 1] = r
    S[:, 0, 3] = f
    S[:, 1, 1] = -t
    S[:, 1, 0] = r
    S[:, 1, 2] = -f
    S[:, 2, 2] = t
    S[:, 2, 3] = r
    S[:, 2, 1] = -f
    S[:, 3, 3] = -t
    S[:, 3, 2] = r
    S[:, 3, 0] = f
    if not isinstance(inputs, np.ndarray):
        inputs = np.array(inputs)
    inputs = np.tile(inputs, (N, 1))
    assert inputs.shape == (N, 4)
    outputs = np.einsum('nij,nj->ni', S, inputs)
    return outputs


# In[ ]:


tp = d[2].ravel()
rp = d[9].ravel()
tau = d[10].ravel()
taup = np.sqrt(np.abs(rp)**2 + np.abs(tau)**2)
fp = d[17].ravel()
o = single_node(inp, tp, rp, fp) # Outputs from RG into S node
oa = single_node(inp, np.abs(tp), np.abs(rp), np.abs(fp)) # Abs outputs from RG into S node
otau = single_node(inp, np.abs(tp), np.abs(taup), np.abs(fp)) # Using r = sqrt(r'^2 + tau'^2)


# In[1107]:


# in the order of the input matrix, outputs are O1_up, O2_up, O1_down, O2_down
print("Number of t'   values that are the same as the output = ", (tp == o[:, 0]).sum())
print("Number of r'   values that are the same as the output = ", (rp == o[:, 1]).sum())
# print("Number of tau' values that are the same as the output = ", (tau == o[:, 2]).sum())
print("Number of f'   values that are the same as the output = ", (fp == o[:, 3]).sum())
print(np.allclose(np.abs(tp)**2 + np.abs(rp)**2 + np.abs(tau)**2 + np.abs(fp)**2, 1.0, atol=1e-12)) # RG outputs sum to 1
print(np.allclose(np.abs(tp)**2 + np.abs(rp)**2 + np.abs(fp)**2, 1.0, atol=1e-12))


# In[ ]:


outsum = np.abs(tp)**2 + np.abs(rp)**2 + np.abs(fp)**2
print(np.allclose(np.abs(o[:, 0])**2 + np.abs(o[:, 1])**2 + np.abs(o[:, 2])**2 + np.abs(o[:, 3])**2, 1, atol=1e-12)) # Saddle node outputs sum to 1
print(np.allclose(np.abs(o[:, 0])**2 + np.abs(o[:, 1])**2 + + np.abs(o[:, 3])**2, outsum, atol=1e-12)) # Saddle node outputs sum to the sum of inputs


# In[997]:


print("Number of t'   values that are the same as the output = ", (np.abs(tp) == oa[:, 0]).sum())
print("Number of r'   values that are the same as the output = ", (np.abs(rp) == oa[:, 1]).sum())
print("Number of tau' values that are the same as the output = ", (np.abs(tau) == oa[:, 2]).sum())
print("Number of f'   values that are the same as the output = ", (np.abs(fp) == oa[:, 3]).sum())


# In[999]:


print(np.allclose(np.abs(otau[:, 0])**2 + np.abs(otau[:, 1])**2 + np.abs(otau[:, 2])**2 + np.abs(otau[:, 3])**2, 1, atol=1e-12))
print(np.allclose(np.abs(tp)**2 + np.abs(fp)**2 + np.abs(taup)**2,1, atol=1e-12))
print("Number of t'   values that are the same as the output = ", (np.abs(tp) == otau[:, 0]).sum())
print("Number of r'   values that are the same as the output = ", (np.abs(taup) == otau[:, 1]).sum())
# print("Number of tau' values that are the same as the output = ", (tau == otau[:, 2]).sum())
print("Number of f'   values that are the same as the output = ", (np.abs(fp) == otau[:, 3]).sum())


# In[ ]:





# # Report stuff

# ## Disorder potential

# In[ ]:


"""
2D Disordered Potential for the Chalker-Coddington Network Model
----------------------------------------------------------------
Generates a smooth random potential V(x,y), identifies its critical points
(maxima, minima, saddle points), and plots the landscape with annotated
topology — consistent with the physical picture in:

  Chalker, J. T. & Coddington, P. D. (1988).
  Percolation, quantum tunnelling and the integer Hall effect.
  J. Phys. C: Solid State Phys., 21, 2665–2679.

In the CC model electrons drift along iso-potential contours and tunnel
quantum-mechanically at saddle-point nodes. The network is therefore
defined entirely by the saddle-point geometry of V(x,y).

Critical-point classification uses the 2×2 Hessian H of V:
  det(H) > 0, tr(H) < 0  →  maximum
  det(H) > 0, tr(H) > 0  →  minimum
  det(H) < 0              →  saddle point
(Morse theory; see e.g. Nakahara, "Geometry, Topology and Physics", §2.4)
"""

# import numpy as np
# import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter
from scipy.signal import argrelextrema

# ── Reproducibility ──────────────────────────────────────────────────────────
RNG = np.random.default_rng(seed=42)

# ── Grid parameters ──────────────────────────────────────────────────────────
N        = 400          # grid points per axis
L        = 1.0          # physical side length
x        = np.linspace(0, L, N)
y        = np.linspace(0, L, N)
X, Y     = np.meshgrid(x, y)
dx       = x[1] - x[0]

# ── Generate smooth random potential ─────────────────────────────────────────
# White noise → Gaussian smoothed to correlation length xi.
# xi controls how many saddle points appear; smaller xi → denser network.
xi_pixels = 30          # correlation length in grid points (~4.5 % of L)
raw       = RNG.standard_normal((N, N))
V         = gaussian_filter(raw, sigma=xi_pixels)
V        /= V.std()     # normalise to unit variance

# ── Numerical gradient and Hessian ───────────────────────────────────────────
Vy, Vx   = np.gradient(V,  dx)          # first derivatives
Vxx, Vxy = np.gradient(Vx, dx)
Vyx, Vyy = np.gradient(Vy, dx)

grad_mag  = np.sqrt(Vx**2 + Vy**2)

# ── Locate critical points ────────────────────────────────────────────────────
# A critical point sits where |∇V| has a local minimum near zero.
# We threshold on the gradient magnitude, then classify by the Hessian.

GRAD_THRESH = 0.12 * grad_mag.max()   # tune to control point density
border      = 15                       # ignore edge artefacts (pixels)

# Binary mask: small gradient
low_grad = (grad_mag < GRAD_THRESH)

# Restrict to interior
mask = np.zeros_like(low_grad, dtype=bool)
mask[border:-border, border:-border] = True
candidates = np.argwhere(low_grad & mask)

# Classify each candidate by det(H) and tr(H)
maxima  = []
minima  = []
saddles = []

# Simple non-maximum-suppression to avoid clustering
MIN_SEP = 2*xi_pixels * dx  # minimum separation in physical units

def _too_close(pt, existing, min_sep):
    if not existing:
        return False
    arr = np.array(existing)
    return np.any(np.linalg.norm(arr - pt, axis=1) < min_sep)

for (j, i) in candidates:
    det_H = Vxx[j, i] * Vyy[j, i] - Vxy[j, i] * Vyx[j, i]
    tr_H  = Vxx[j, i] + Vyy[j, i]
    pt    = np.array([x[i], y[j]])

    if det_H > 0 and tr_H < 0:
        if not _too_close(pt, maxima, MIN_SEP):
            maxima.append(pt)
    elif det_H > 0 and tr_H > 0:
        if not _too_close(pt, minima, MIN_SEP):
            minima.append(pt)
    elif det_H < 0:
        if not _too_close(pt, saddles, MIN_SEP):
            if np.abs(V[j, i]) < 0.5:
                saddles.append(pt)

maxima  = np.array(maxima)  if maxima  else np.empty((0, 2))
minima  = np.array(minima)  if minima  else np.empty((0, 2))
saddles = np.array(saddles) if saddles else np.empty((0, 2))

print(f"Critical points found — maxima: {len(maxima)}, "
      f"minima: {len(minima)}, saddles: {len(saddles)}")

# ── Plotting ──────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 7))

# Filled-contour background
cf = ax.contourf(X, Y, V, levels=60,
                 cmap="RdBu_r", alpha=0.85)

# Iso-potential lines — the CC drift trajectories
n_iso = 30
iso_levels = np.linspace(V.min(), V.max(), n_iso)
cs = ax.contour(X, Y, V, levels=iso_levels,
                colors="k", linewidths=0.35, alpha=0.45)

ax.contour(X, Y, V, levels=[0.0], colors="k", linestyles="--", alpha=0.5, linewidths=1.5)
# ── Colour bar ───────────────────────────────────────────────────────────────
cbar = fig.colorbar(cf, ax=ax, pad=0.02, fraction=0.046)
cbar.set_label("V(x, y)", fontsize=11)
cbar.ax.tick_params(labelsize=9)

# ── Annotate critical points ─────────────────────────────────────────────────
MARKER_KW = dict(zorder=5, clip_on=False)

# if len(maxima):
#     ax.scatter(maxima[:, 0], maxima[:, 1],
#                marker="^", s=90, color="#FF4444",
#                edgecolors="white", linewidths=0.7,
#                label=f"Maxima ({len(maxima)})", **MARKER_KW)

# if len(minima):
#     ax.scatter(minima[:, 0], minima[:, 1],
#                marker="v", s=90, color="#4488FF",
#                edgecolors="white", linewidths=0.7,
#                label=f"Minima ({len(minima)})", **MARKER_KW)

# if len(saddles):
#     ax.scatter(saddles[:, 0], saddles[:, 1],
#                marker="X", s=90, color="#FFD700",
#                edgecolors="black", linewidths=0.7,
#                label=f"Saddle nodes ({len(saddles)})", **MARKER_KW)

# print(saddles.shape)
if len(saddles):
    for point in range(saddles.shape[0]):
        sx = saddles[point, 0]
        sy = saddles[point, 1]
        circle = plt.Circle((sx, sy), radius=0.027,  # tune radius to your axis scale
                            fill=False, color='green',
                            linewidth=1.5, linestyle='-')
        ax.add_patch(circle)

# ── Labels and legend ────────────────────────────────────────────────────────
# ax.set_xlabel("x  [arb. units]", fontsize=12)
# ax.set_ylabel("y  [arb. units]", fontsize=12)
# ax.set_title("Disordered Potential Landscape\n"
#              "Chalker–Coddington Network Model",
#              fontsize=13, fontweight="bold")
ax.set_aspect("equal")
ax.tick_params(labelsize=9)

# legend = ax.legend(loc="upper right", fontsize=9,
#                    framealpha=0.85, edgecolor="grey")

# Small annotation explaining the physics
note = ("Iso-potential contours (black lines) are\n"
        "the electron drift trajectories; tunnelling\n"
        "occurs quantum-mechanically at saddle nodes.")
# ax.text(0.02, 0.02, note, transform=ax.transAxes,
#         fontsize=7.5, va="bottom", ha="left",
#         bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.75, ec="grey"))

plt.tight_layout()
# plt.savefig("./report/chalker_coddington.pdf",
#             dpi=180, bbox_inches="tight")
plt.show()
print("Saved → chalker_coddington.pdf")


# In[209]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label, minimum_filter
RNG = np.random.default_rng(seed=7)

N = 500
L = 1.0
x = np.linspace(0, L, N)
y = np.linspace(0, L, N)
X, Y = np.meshgrid(x, y)

# Place alternating maxima/minima on a perturbed checkerboard
# This guarantees saddle points between neighbours at approximately V=0
n_peaks = 6         # n_peaks x n_peaks grid of extrema
sigma   = 0.05      # width of each Gaussian — controls overlap
jitter  = 0.01       # random displacement to break perfect regularity

V = np.zeros((N, N))
saddle_hints = []    # approximate saddle locations for annotation

for ix in range(n_peaks):
    for iy in range(n_peaks):
        # Centre of this extremum
        cx = (ix + 0.5) / n_peaks + RNG.uniform(-jitter, jitter)
        cy = (iy + 0.5) / n_peaks + RNG.uniform(-jitter, jitter)
        cx = np.clip(cx, 0.08, 0.92)
        cy = np.clip(cy, 0.08, 0.92)

        # Checkerboard sign: +1 = maximum, -1 = minimum
        sign = 1 if (ix + iy) % 2 == 0 else -1
        amplitude = sign * (1.0 + 0.3 * RNG.standard_normal())

        V += amplitude * np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2))

# Normalise to unit std
V /= V.std()

# ── Find saddle points numerically ───────────────────────────────────────────
dx_   = x[1] - x[0]
Vy_, Vx_   = np.gradient(V,  dx_)
Vxx_, Vxy_ = np.gradient(Vx_, dx_)
Vyx_, Vyy_ = np.gradient(Vy_, dx_)
grad_mag_  = np.sqrt(Vx_**2 + Vy_**2)

GRAD_THRESH = 0.08 * grad_mag_.max()
# V_EQ_THRESH = 0.15          # saddle points should sit near V=0
MIN_SEP     = 0.05          # minimum physical separation between reported saddles
border      = 20

# candidates = np.argwhere(
#     (grad_mag_ < GRAD_THRESH) &
#     (np.abs(V)  < V_EQ_THRESH) &
#     np.pad(np.ones((N-2*border, N-2*border), dtype=bool),
#            border, constant_values=False)
# )

# saddles = []
# for (j, i) in candidates:
#     det_H = Vxx_[j,i]*Vyy_[j,i] - Vxy_[j,i]*Vyx_[j,i]
#     if det_H < 0:
#         pt = np.array([x[i], y[j]])
#         if not any(np.linalg.norm(pt - s) < MIN_SEP for s in saddles):
#             saddles.append(pt)

# ── Restrict to near-equipotential strip ──────────────────────────────────
V_EQ_THRESH = 0.20
eq_strip = np.abs(V) < V_EQ_THRESH

# Mask gradient magnitude outside the strip
grad_in_strip = np.where(eq_strip, grad_mag_, np.inf)

# Find local minima of gradient magnitude within the strip
# A local minimum means gradient is smallest in its neighbourhood
footprint_size = int(0.1 * MIN_SEP / dx_)   # neighbourhood radius in pixels
footprint_size = max(footprint_size, 5)

local_min = (grad_in_strip == minimum_filter(
                 grad_in_strip, size=footprint_size))

# Further threshold: only keep if gradient is below a local percentile
# within the strip (not global)
strip_grads = grad_mag_[eq_strip]
local_thresh = np.percentile(strip_grads, 25)  # bottom 15% within strip

candidates = np.argwhere(
    local_min &
    eq_strip &
    (grad_mag_ < local_thresh) &
    np.pad(np.ones((N-2*border, N-2*border), dtype=bool),
           border, constant_values=False)
)

# Classify and deduplicate as before
saddles = [(0.18, 0.49), (0.33, 0.66), (0.5, 0.67), (0.5, 0.17), (0.67, 0.17), (0.83, 0.33)]
# for (j, i) in candidates:
#     det_H = Vxx_[j,i]*Vyy_[j,i] - Vxy_[j,i]*Vyx_[j,i]
#     if det_H < 0:
#         pt = np.array([x[i], y[j]])
#         if not any(np.linalg.norm(pt - s) < MIN_SEP for s in saddles):
#             saddles.append(pt)
saddles = np.array(saddles) if saddles else np.empty((0,2))
print(f"Saddle points found: {len(saddles)}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6.5))

cf = ax.contourf(X, Y, V, levels=60, cmap='RdBu_r', alpha=0.9)
ax.contour(X, Y, V,
           levels=np.linspace(V.min(), V.max(), 21),
           colors='k', linewidths=0.3, alpha=0.4)
ax.contour(X, Y, V, levels=[0.0],
           colors='g', linestyles='--', linewidths=1.6, alpha=0.7)

cbar = fig.colorbar(cf, ax=ax, pad=0.02, fraction=0.046)
cbar.set_label('V(x, y)', fontsize=12)

# Draw circles at saddle points
r = 0.05

for sp in saddles:
    ax.add_patch(plt.Circle(sp, radius=r,
                             fill=False, color='black',
                             linewidth=2.0, zorder=6))

ax.set_aspect('equal')
ax.tick_params(labelsize=12)
plt.tight_layout()
plt.savefig('./report/cc_potential.png', dpi=180, bbox_inches='tight')
plt.show()


# In[224]:


fig, ax = plt.subplots(figsize=(3.5, 2.2))
ax.axis('off')

# ── First equation ────────────────────────────────────────────────
# Left bracket + entries
ax.text(0.03, 0.70, '(',   fontsize=32, ha='center', va='center', transform=ax.transAxes)
ax.text(0.10, 0.78, r'$O_1$', fontsize=13, ha='center', va='center', transform=ax.transAxes)
ax.text(0.10, 0.62, r'$O_2$', fontsize=13, ha='center', va='center', transform=ax.transAxes)
ax.text(0.17, 0.70, ')',   fontsize=32, ha='center', va='center', transform=ax.transAxes)

ax.text(0.27, 0.70, r'$= S \cdot$', fontsize=13, ha='center', va='center', transform=ax.transAxes)

ax.text(0.38, 0.70, '(',   fontsize=32, ha='center', va='center', transform=ax.transAxes)
ax.text(0.45, 0.78, r'$I_1$', fontsize=13, ha='center', va='center', transform=ax.transAxes)
ax.text(0.45, 0.62, r'$I_2$', fontsize=13, ha='center', va='center', transform=ax.transAxes)
ax.text(0.52, 0.70, ')',   fontsize=32, ha='center', va='center', transform=ax.transAxes)

# ── Second equation ───────────────────────────────────────────────
ax.text(0.1, 0.28, r'$S =$', fontsize=13, ha='center', va='center', transform=ax.transAxes)

ax.text(0.18, 0.28, '(',   fontsize=32, ha='center', va='center', transform=ax.transAxes)
ax.text(0.27, 0.36, r'$t$',  fontsize=13, ha='center', va='center', transform=ax.transAxes)
ax.text(0.38, 0.36, r'$r$',  fontsize=13, ha='center', va='center', transform=ax.transAxes)
ax.text(0.27, 0.20, r'$r$',  fontsize=13, ha='center', va='center', transform=ax.transAxes)
ax.text(0.38, 0.20, r'$-t$', fontsize=13, ha='center', va='center', transform=ax.transAxes)
ax.text(0.46, 0.28, ')',   fontsize=32, ha='center', va='center', transform=ax.transAxes)
plt.tight_layout()
plt.savefig('./report/smatrix_equations.png', dpi=300, transparent=True,
            bbox_inches='tight', pad_inches=0.1)

plt.show()


# ## Landau and Conductance

# In[30]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

# ═══════════════════════════════════════════════════════════════════
# FIGURE 1 — Landau Bands (DOS) for IQHE and QSHE
# Reference: Obuse et al. Fig. 2
# ═══════════════════════════════════════════════════════════════════

def gaussian(E, E0, sigma, A=1.0):
    return A * np.exp(-0.5 * ((E - E0) / sigma)**2)

E = np.linspace(-0.5, 4.5, 2000)

# Landau level centres
E_centres = [0.5, 1.5, 2.5, 3.5]
sigma_broad = 0.18      # broadened width
sigma_ext   = 0.045     # extended-state width (narrow peak at centre)

fig, axes = plt.subplots(1, 2, figsize=(7, 5), sharey=True)
fig.subplots_adjust(wspace=0.05)

titles = ['IQHE', 'QSHE']

for col, ax in enumerate(axes):
    DOS_total = np.zeros_like(E)
    DOS_loc   = np.zeros_like(E)
    DOS_ext   = np.zeros_like(E)

    for E0 in E_centres:
        band = gaussian(E, E0, sigma_broad)
        DOS_total += band

        if col == 0:
            # IQHE: narrow extended-state peak at centre
            ext = gaussian(E, E0, sigma_ext, A=0.55)
            DOS_ext += ext
            DOS_loc += (band - ext)
        else:
            # QSHE: broadened band, entire band is extended (grey)
            # thin black line marks critical energies at band edges
            DOS_ext += band

    # Plot localised (white/hatched) and extended (grey) regions
    if col == 0:
        ax.fill_betweenx(E, DOS_loc, 0,
                         color='white', linewidth=0)
        ax.fill_betweenx(E, DOS_ext, 0,
                         color='#cc2222', linewidth=0,
                         label='extended')
        ax.plot(DOS_total, E, 'k-', linewidth=1.2)

        # Mark critical energies with thin horizontal black lines
        for E0 in E_centres:
            ax.axhline(E0, color='black', linewidth=0.8,
                       xmin=0.0, xmax=0.18, linestyle='-')

    else:
        # QSHE: grey = extended, rest = localised (white)
        ax.fill_betweenx(E, DOS_total, 0,
                         color='#999999', linewidth=0,
                         label='extended')
        ax.plot(DOS_total, E, 'k-', linewidth=1.2)

        # Critical energies at band edges (where localised→extended transition)
        for E0 in E_centres:
            for edge in [E0 - 0.5*sigma_broad*3.5, E0 + 0.5*sigma_broad*3.5]:
                ax.axhline(edge, color='black', linewidth=0.9,
                           xmin=0.0, xmax=0.25, linestyle='-')

    ax.set_xlim(-0.05, 1.3)
    ax.set_ylim(-0.3, 4.8)
    ax.set_xlabel('DOS', fontsize=12)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_title(f'({chr(97+col)}) {titles[col]}', fontsize=12, loc='left')

    # Label Landau level energies on y-axis for first panel
    if col == 0:
        for n, E0 in enumerate(E_centres):
            ax.text(-0.04, E0, f'$E_{{{n}}}$',
                    ha='right', va='center', fontsize=10,
                    transform=ax.get_yaxis_transform())

# Shared y-axis label
axes[0].set_ylabel('Chemical potential  $\\mu$', fontsize=12)

# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#cc2222', label='Extended'),
                   Patch(facecolor='white', edgecolor='black', label='Localised')]
fig.legend(handles=legend_elements, loc='upper center',
           ncol=2, fontsize=10, frameon=True,
           bbox_to_anchor=(0.5, 0.97))

plt.suptitle('Density of States — Landau bands', fontsize=12, y=1.01)
plt.tight_layout()
# plt.savefig('./report/landau_bands.pdf', dpi=180, bbox_inches='tight')
plt.show()


# ═══════════════════════════════════════════════════════════════════
# FIGURE 2 — Conductance plateaus (σ_xy and σ_xx vs E_F)
# Reference: Cain & Roemer Fig. 2.4
# ═══════════════════════════════════════════════════════════════════

n_landau  = 4
E_centres = np.array([0.5, 1.5, 2.5, 3.5])
sigma_b   = 0.18
E_plot    = np.linspace(0.0, 4.2, 3000)

# σ_xy: step function, rising through each band centre
# Modelled as sum of sigmoids
def sigmoid(E, E0, width=0.06):
    return 1.0 / (1.0 + np.exp(-(E - E0) / width))

sigma_xy = np.zeros_like(E_plot)
for E0 in E_centres:
    sigma_xy += sigmoid(E_plot, E0, width=0.055)

# σ_xx: Gaussian peak at each band centre (non-zero only during transition)
sigma_xx = np.zeros_like(E_plot)
for E0 in E_centres:
    sigma_xx += gaussian(E_plot, E0, sigma_b * 0.55, A=0.85)

# DOS underneath for reference
DOS = np.zeros_like(E_plot)
for E0 in E_centres:
    DOS += gaussian(E_plot, E0, sigma_b)

fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(12, 6),
                                      sharex=True,
                                      gridspec_kw={'hspace': 0.08})
# fig, ax_top = plt.subplots(figsize=(6, 6),
#                                       gridspec_kw={'hspace': 0.08})

# ── Top panel: σ_xy (dashed) and σ_xx (solid) ──────────────────────
ax_top.plot(E_plot, sigma_xy, 'k--', linewidth=1.4, label=r'$\sigma_{xy}$')
ax_top.plot(E_plot, sigma_xx, 'k-',  linewidth=1.4, label=r'$\sigma_{xx}$')

# y-axis ticks at integer plateau values
n_ticks = n_landau + 1
ax_top.set_yticks(range(n_ticks))
ax_top.set_yticklabels([f'$n={i}$' for i in range(n_ticks)], fontsize=14)
ax_top.set_ylabel(r'$\sigma \ [e^2/h]$', fontsize=14)
ax_top.set_ylim(-0.2, n_landau + 0.4)
ax_top.legend(loc='upper left', fontsize=14, frameon=False)
ax_top.spines['bottom'].set_visible(False)
ax_top.tick_params(bottom=False)

# Horizontal dashed guide lines at each plateau
for n in range(n_ticks):
    ax_top.axhline(n, color='grey', linewidth=0.4, linestyle=':', zorder=0)

# ── Bottom panel: DOS with extended (red) and localised (white) ────
ax_bot.fill_between(E_plot, DOS, color='white',
                    edgecolor='black', linewidth=1.0,
                    label='Localised')

for E0 in E_centres:
    # Shade narrow central region red = extended states
    ext = gaussian(E_plot, E0, sigma_b * 0.1, A=1.0)
    mask = ext > 0.05 * ext.max()
    ax_bot.fill_between(E_plot, DOS,
                         where=mask,
                         color='#cc2222', linewidth=0,
                         label='Extended' if E0 == E_centres[0] else '')
    # Mark E_n
    ax_bot.axvline(E0, color='k', linewidth=0.6, linestyle='-',
                   ymin=0, ymax=0.06)
    ax_bot.text(E0, -0.30 * DOS.max(), f'$E_{{{int(E0-0.5)}}}$',
                ha='center', va='top', fontsize=14)

ax_bot.plot(E_plot, DOS, 'k-', linewidth=1.1)
ax_bot.set_ylabel('DOS', fontsize=14)
ax_bot.set_xlabel(r'$E_F$', fontsize=14, labelpad=10)
ax_bot.set_ylim(-0.15, 1.3 * DOS.max())
ax_bot.set_xlim(0.0, 4.2)
ax_bot.set_yticks([])
ax_bot.spines['top'].set_visible(False)

handles = [mpatches.Patch(facecolor='#cc2222', label='Extended'),
           mpatches.Patch(facecolor='white', edgecolor='black',
                          label='Localised')]
ax_bot.legend(handles=handles, fontsize=12, frameon=False,
              loc='upper right')

plt.savefig('./report/conductance_plateaus.pdf', dpi=180, bbox_inches='tight')
plt.show()
close_old_plots()


# In[42]:


"""
Figures for IQHE and QSHE:
  1. Disorder-broadened Landau bands (DOS) — IQHE and QSHE comparison
  2. Conductance plateaus — σ_xy and σ_xx vs Fermi energy

Physical content references:
  - Landau level broadening by disorder and localisation structure:
      B. Kramer & A. MacKinnon, Rep. Prog. Phys. 56 (1993) 1469
  - IQHE conductance quantisation:
      K. von Klitzing, G. Dorda & M. Pepper, PRL 45 (1980) 494
  - Extended states at Landau band centres (IQHE):
      B. Huckestein, Rev. Mod. Phys. 67 (1995) 357
  - QSHE DOS structure (extended throughout band, metallic phase):
      H. Obuse et al., PRB 76 (2007) 075301  [Fig. 2 therein]
  - Symplectic class: weak anti-localisation → metallic phase:
      S. Hikami, A. Larkin & Y. Nagaoka, Prog. Theor. Phys. 63 (1980) 707
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from matplotlib import rcParams

# ── Global style ─────────────────────────────────────────────────────────────
rcParams.update({
    'font.family'      : 'serif',
    'mathtext.fontset' : 'cm',
    'font.size'        : 11,
    'axes.linewidth'   : 1.1,
    'xtick.direction'  : 'in',
    'ytick.direction'  : 'in',
})

def gauss(E, E0, s, A=1.0):
    return A * np.exp(-0.5 * ((E - E0) / s)**2)

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Landau band DOS: IQHE vs QSHE
# ═══════════════════════════════════════════════════════════════════════════
# Physical picture
# ----------------
# IQHE (unitary class): disorder broadens each sharp Landau level into a band.
# All states are Anderson-localised EXCEPT a single extended state at the band
# centre E_n. This is the state responsible for the plateau-to-plateau
# transition. (Huckestein 1995; Kramer & MacKinnon 1993)
#
# QSHE (symplectic class): time-reversal symmetry prevents backscattering
# (weak anti-localisation). In the presence of disorder the system passes
# through a *metallic* phase between the two insulating phases; the entire
# Landau band can host extended states. (Obuse et al. 2007, Fig. 2d)
# ═══════════════════════════════════════════════════════════════════════════

E_centres = np.array([1.0, 2.0, 3.0, 4.0])   # Landau level energies
n_LL      = len(E_centres)
sigma_b   = 0.22    # disorder broadening (half-width)
sigma_ext = 0.04    # width of extended-state spike (IQHE only)

mu = np.linspace(0.2, 4.8, 4000)   # chemical potential axis (vertical)

# Compute DOS components
DOS_total   = sum(gauss(mu, E0, sigma_b) for E0 in E_centres)
DOS_ext_IQH = sum(gauss(mu, E0, sigma_ext, A=0.55) for E0 in E_centres)
DOS_loc_IQH = DOS_total - DOS_ext_IQH

# QSHE: whole band is extended (grey), with a slightly narrower peak to
# reflect the reduced broadening from weak anti-localisation
DOS_QSH = sum(gauss(mu, E0, sigma_b * 0.85) for E0 in E_centres)

# ── Colour palette ───────────────────────────────────────────────────────────
C_EXT  = '#C0392B'   # deep red   — extended states
C_LOC  = '#ECF0F1'   # near-white — localised states
C_MET  = '#5D6D7E'   # steel grey — metallic/extended (QSHE)
C_CRIT = '#2C3E50'   # dark ink

fig1, (ax_i, ax_q) = plt.subplots(
    1, 2, figsize=(8, 6), sharey=True,
    gridspec_kw={'wspace': 0.04}
)

# ── Helper: shade a horizontal DOS curve with two-layer fill ─────────────────
def shade_dos(ax, dos_loc, dos_ext, mu, colour_loc, colour_ext):
    ax.fill_betweenx(mu, dos_loc + dos_ext, color=colour_loc, linewidth=0)
    # ax.fill_betweenx(mu, dos_ext,           color=colour_ext, linewidth=0, zorder=2)
    ax.plot(dos_loc + dos_ext, mu, color=C_CRIT, linewidth=1.15, zorder=3)

# ── IQHE panel ───────────────────────────────────────────────────────────────
shade_dos(ax_i, DOS_loc_IQH, DOS_ext_IQH, mu, C_LOC, C_EXT)

# Thin horizontal lines at E_n marking critical (extended) states
for E0 in E_centres:
    ax_i.axhline(E0, color=C_EXT, linewidth=2.8, linestyle='-',
                 xmin=0.0, xmax=0.74, zorder=4)

ax_i.set_title('IQHE', fontsize=12, fontweight='bold', pad=6)
ax_i.set_xlabel('DOS  (arb. units)', fontsize=12)
ax_i.set_ylabel(r'Fermi Energy  $E_F$', fontsize=14, labelpad=25)

# ── QSHE panel ───────────────────────────────────────────────────────────────
# Entire band is extended (metallic phase) — fill uniformly grey
ax_q.fill_betweenx(mu, DOS_QSH, color=C_LOC, linewidth=0, alpha=0.75, zorder=1)
ax_q.plot(DOS_QSH, mu, color=C_CRIT, linewidth=1.15, zorder=3)

# Critical energies at band *edges* (not centres) — metal–insulator transition
# The QSHE has transitions at both edges of the broadened band
for E0 in E_centres:
    for de in [-0.9 * sigma_b, + 0.9 * sigma_b]:
        ax_q.axhline(E0 + de, color=C_CRIT, linewidth=2, linestyle='-',
                     xmin=0.0, xmax=0.4, zorder=4)
        bandmask = (mu >= E0 - de) & (mu <= E0+de)
        ax_q.fill_betweenx(mu, DOS_QSH, where = bandmask, color=C_MET, linewidth=0, alpha=0.75, zorder=2)

ax_q.set_title('QSHE', fontsize=12, fontweight='bold', pad=6)
ax_q.set_xlabel('DOS  (arb. units)', fontsize=12)

# ── Shared formatting ────────────────────────────────────────────────────────
for ax, n_label in [(ax_i, True), (ax_q, False)]:
    ax.set_xlim(-0.04, 1.35)
    ax.set_ylim(0.3, 5.1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if n_label:
        for n, E0 in enumerate(E_centres):
            ax.text(-0.035, E0, f'$E_{n}$',
                    ha='right', va='center', fontsize=14,
                    transform=ax.get_yaxis_transform())

# ── Legend ───────────────────────────────────────────────────────────────────
from matplotlib.patches import Patch
leg_elems = [
    Patch(facecolor=C_EXT, edgecolor='none', label='Extended states'),
    Patch(facecolor=C_LOC, edgecolor=C_CRIT, linewidth=0.7,
          label='Localised states'),
    Patch(facecolor=C_MET, edgecolor='none', alpha=0.75,
          label='Metallic (extended)'),
]
fig1.legend(handles=leg_elems, loc='lower center', ncol=3,
            fontsize=14, frameon=True, framealpha=0.9,
            edgecolor='#bbb',
            bbox_to_anchor=(0.5, -0.01))

fig1.suptitle('Disorder-broadened Landau bands', fontsize=14,
              fontweight='bold', y=1.01)
# fig1.tight_layout()
fig1.savefig('./report/landau_bands.png',
             dpi=180, bbox_inches='tight')
fig1.savefig('./report/landau_bands.pdf',
             dpi=180, bbox_inches='tight')
print("Saved landau_bands.pdf")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Conductance plateaus σ_xy and σ_xx vs E_F
# ═══════════════════════════════════════════════════════════════════════════
# σ_xy = n e²/h in the plateau regions (Fermi level in localised states).
# σ_xx ≠ 0 only when E_F crosses the band centre (extended states allow
# dissipative transport). Both quantities are schematic; the staircase in
# σ_xy is modelled with a sum of sigmoids, and σ_xx with Gaussians.
# (von Klitzing et al. 1980; Huckestein 1995 §II)
# ═══════════════════════════════════════════════════════════════════════════

# def sigmoid(E, E0, w=0.05):
#     return 1.0 / (1.0 + np.exp(-(E - E0) / w))

# EF = np.linspace(0.3, 4.7, 5000)

# sig_xy = sum(sigmoid(EF, E0, w=0.05)  for E0 in E_centres)
# sig_xx = sum(gauss(EF, E0, sigma_b * 0.55, A=0.92) for E0 in E_centres)
# dos_bg = sum(gauss(EF, E0, sigma_b)   for E0 in E_centres)

# fig2, axes = plt.subplots(
#     3, 1, figsize=(6.5, 7.5), sharex=True,
#     gridspec_kw={'hspace': 0.08, 'height_ratios': [2, 2, 1.4]}
# )
# ax_xy, ax_xx, ax_dos = axes

# # ── σ_xy ─────────────────────────────────────────────────────────────────────
# # Background: shade plateau regions (Fermi level in localised states) lightly
# for n, E0 in enumerate(E_centres):
#     E_lo = E0 - 2.6 * sigma_b
#     E_hi = E0 + 2.6 * sigma_b
#     # Shade between adjacent band edges as plateau regions
#     if n == 0:
#         ax_xy.axvspan(EF[0], E_lo, color='#EBF5FB', zorder=0)
#     ax_xy.axvspan(E_hi,
#                   E_centres[n+1] - 2.6*sigma_b if n < n_LL-1 else EF[-1],
#                   color='#EBF5FB', zorder=0)

# ax_xy.plot(EF, sig_xy, color='#1A252F', linewidth=2.0, zorder=3,
#            label=r'$\sigma_{xy}$')

# # Integer plateau labels on right y-axis
# for n in range(n_LL + 1):
#     ax_xy.axhline(n, color='#BDC3C7', linewidth=0.6, zorder=1)
# ax_xy.set_yticks(range(n_LL + 1))
# ax_xy.set_yticklabels([f'${n}\\ e^2/h$' for n in range(n_LL + 1)],
#                        fontsize=8)
# ax_xy.set_ylim(-0.15, n_LL + 0.3)
# ax_xy.set_ylabel(r'$\sigma_{xy}$', fontsize=13)
# ax_xy.legend(loc='upper left', fontsize=10, frameon=False)
# ax_xy.spines['bottom'].set_visible(False)
# ax_xy.tick_params(bottom=False)

# # ── σ_xx ─────────────────────────────────────────────────────────────────────
# ax_xx.fill_between(EF, sig_xx, color='#AED6F1', alpha=0.55, zorder=1)
# ax_xx.plot(EF, sig_xx, color='#1A5276', linewidth=1.8, zorder=2,
#            label=r'$\sigma_{xx}$')
# ax_xx.set_yticks([0, 0.5, 1.0])
# ax_xx.set_yticklabels(['$0$', '', r'$e^2/h$'], fontsize=9)
# ax_xx.set_ylim(-0.05, 1.25)
# ax_xx.set_ylabel(r'$\sigma_{xx}$', fontsize=13)
# ax_xx.legend(loc='upper right', fontsize=10, frameon=False)
# ax_xx.spines['bottom'].set_visible(False)
# ax_xx.tick_params(bottom=False)

# # ── DOS background ────────────────────────────────────────────────────────────
# # Shade extended-state cores red
# dos_ext_plt = sum(gauss(EF, E0, sigma_ext, A=0.55) for E0 in E_centres)
# # print(dos_ext_plt)
# # ax_dos.fill_between(EF, dos_bg,color=C_LOC, edgecolor='none')
# # ax_dos.fill_between(EF, dos_ext_plt,  color=C_EXT, alpha=0.85, zorder=2,
# #                     label='Extended')
# # ax_dos.plot(EF, dos_bg, color=C_CRIT, linewidth=1.1, zorder=3)
# for E0 in E_centres:
#     ext = gaussian(EF, E0, sigma_ext, A=0.55)
#     mask = ext > 0.05 * ext.max()
#     ax_dos.fill_between(EF, ext,
#                             where=mask,
#                             color='#cc2222', linewidth=0,
#                             label='Extended')
#     ax_dos.plot(EF, dos_bg, color=C_CRIT, linewidth=1.1, zorder=3)
#     ax_dos.fill_between(EF, dos_bg,color=C_LOC, edgecolor='none')

# for n, E0 in enumerate(E_centres):
#     ax_dos.axvline(E0, color='#7F8C8D', linewidth=0.7, linestyle='--', zorder=1)
#     ax_dos.text(E0, -0.4 * dos_bg.max(), f'$E_{n}$',
#                 ha='center', va='top', fontsize=9)

# ax_dos.set_ylabel('DOS', fontsize=11)
# ax_dos.set_xlabel(r'Fermi energy  $E_F$', fontsize=12, labelpad=12)
# ax_dos.set_ylim(-0.18 * dos_bg.max(), 1.3 * dos_bg.max())
# ax_dos.set_yticks([])
# ax_dos.set_xlim(EF[0], EF[-1])
# ax_dos.spines['top'].set_visible(False)

# # Remove x-ticks on upper panels
# for ax in [ax_xy, ax_xx]:
#     ax.set_xlim(EF[0], EF[-1])
#     ax.spines['top'].set_visible(False)

# # Vertical lines through all panels at E_n
# for E0 in E_centres:
#     for ax in [ax_xy, ax_xx]:
#         ax.axvline(E0, color='#BDC3C7', linewidth=0.7,
#                    linestyle='--', zorder=1)

# fig2.suptitle('Conductance plateaus — IQHE', fontsize=13,
#               fontweight='bold', y=1.005)
# # fig2.tight_layout()
# # fig2.savefig('/mnt/user-data/outputs/conductance_plateaus.pdf',
# #              dpi=180, bbox_inches='tight')
# print("Saved conductance_plateaus.pdf")
# plt.show()


# In[ ]:




