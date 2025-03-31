"""
This code is used to calculate the yield for low energy runs
with small numbers of counts in the region of interest.
It works as follows : 
    - beam and data file paths for each target of interest are provided in the code function
    with the variables MPA_FILES_PATTERN and BEAM_FILES_PATTERN.
    -The energy range for the counts to be counted in is defined by
    the energy_range variable and is determined as stated in the report.
    - Excluded runs are defined using the RUNS_TO _EXLUDE index.
    -If the fit is chosen to have to go through the first data point the 
    FORCE_LINEAR_FIT variable is set to True, if it is to be free it is set
    to False
    -
    
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob

RUNS_TO_EXCLUDE = []
FORCE_LINEAR_FIT = True

def load_mpa_file(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
        target_xtitle = "1C"
        data_start = None
        for i, line in enumerate(lines):
            if line.startswith("XTITLE") and target_xtitle in line:
                data_start = i
                break
        for i in range(data_start, len(lines)):
            if lines[i].startswith("[DATA]"):
                data_start = i + 1
                break
        data = []
        for line in lines[data_start:]:
            line = line.strip()
            if not line:
                continue
            try:
                channel, count = map(int, line.split())
                data.append((channel, count))
            except ValueError:
                break
        return pd.DataFrame(data, columns=["Channel", "Counts"])
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def sum_counts_in_roi(df, slope, intercept, energy_range):
    if df is None or df.empty:
        return 0
    df["Energy"] = slope * df["Channel"] + intercept
    e_min, e_max = energy_range
    mask = (df["Energy"] >= e_min) & (df["Energy"] <= e_max)
    return df.loc[mask, "Counts"].sum()

def load_beam_file(file_path):
    time_vals = []
    charge_vals = []
    try:
        with open(file_path, 'r') as bf:
            for line in bf:
                line = line.strip().rstrip(',')
                if not line:
                    continue
                parts = line.split(',')
                if len(parts) < 2:
                    continue
                try:
                    t_ = float(parts[0])
                    c_ = float(parts[1])
                    time_vals.append(t_)
                    charge_vals.append(c_)
                except ValueError:
                    continue
    except Exception as e:
        print(f"Error loading beam file {file_path}: {e}")
        return [], []
    return time_vals, charge_vals

def compute_and_plot_yields():
    mpa_files_pattern = r"C:\SH Data Files\Data Files\Target #7 (399.74)\10B_a_targ_test_*.mpa"
    beam_files_pattern = r"C:\SH Data Files\Beam Files\Target #7 (399.74)\10B_a_targ_test*.dat"
    mpa_files = sorted(glob.glob(mpa_files_pattern))
    beam_info_files = sorted(glob.glob(beam_files_pattern))
    slope = 0.001847
    intercept = 0.613362
    energy_range = (3.33, 3.62)
    cumulative_charge = 0.0
    valid_data_points = []
    valid_errors = []
    exclude_runs = RUNS_TO_EXCLUDE
    for i, data_file in enumerate(mpa_files):
        if i >= len(beam_info_files):
            print(f"No matching beam info for {data_file} (index {i}). Skipping.")
            continue
        df = load_mpa_file(data_file)
        roi_counts = sum_counts_in_roi(df, slope, intercept, energy_range)
        beam_file = beam_info_files[i]
        time_vals, charge_vals = load_beam_file(beam_file)
        if not time_vals or not charge_vals:
            print(f"WARNING: Beam file {beam_file} has no valid data. Skipping run.")
            continue
        this_run_charge = np.sum(charge_vals)
        cumulative_charge += this_run_charge
        if roi_counts == 0:
            print(f"Run {i}: ROI counts = 0 -> skipping yield, but accumulated charge is now {cumulative_charge}.")
            continue
        run_yield = roi_counts / this_run_charge
        point_error = np.sqrt(roi_counts) / this_run_charge
        print(point_error)
        print(run_yield)
        if i in exclude_runs:
            print(f"Run {i} is excluded from the yield plot. Yield is not added, but its charge is accumulated.")
        else:
            valid_data_points.append((cumulative_charge, run_yield))
            valid_errors.append(point_error)
        print(f"Run {i}, MPA file: {data_file}\n"
              f"  -> ROI counts: {roi_counts}\n"
              f"  -> Beam file: {beam_file}\n"
              f"  -> This-run charge: {this_run_charge}\n"
              f"  -> Cumulative charge so far: {cumulative_charge}\n"
              f"  -> Yield: {run_yield}\n")
        plt.figure(figsize=(10, 4))
        plt.plot(time_vals, charge_vals, 'b-', marker='o', markersize=3, label='Beam Current')
        plt.xlabel("Time (s)")
        plt.ylabel("Charge (arbitrary units)")
        plt.title(f"Beam Current vs. Time ({beam_file})")
        plt.grid(True)
        plt.legend()
        plt.show()
        df["Energy"] = df["Channel"] * slope + intercept
        df_roi = df[(df["Energy"] >= energy_range[0]) & (df["Energy"] <= energy_range[1])]
        energy_bins = np.arange(energy_range[0] - slope/2, energy_range[1] + slope/2, slope)
        plt.figure(figsize=(10, 4))
        plt.hist(df_roi["Energy"], bins=energy_bins, weights=df_roi["Counts"], alpha=0.7, 
                 color='blue', label="Counts vs Energy")
        plt.axvspan(energy_range[0], energy_range[1], color='red', alpha=0.3, label=f"ROI: {energy_range}")
        plt.xlim(energy_range)
        plt.xlabel("Energy (MeV)")
        plt.ylabel("Counts")
        plt.title(f"Energy Spectrum ({data_file})")
        plt.grid(True)
        plt.legend()
        plt.show()
    if not valid_data_points:
        print("No valid yields were computed (all had 0 ROI counts, invalid charge, or were excluded).")
        return
    cumulative_charges, raw_yields = zip(*valid_data_points)
    cumulative_charges = np.array(cumulative_charges) * 1e-8
    raw_yields = np.array(raw_yields)
    valid_errs = np.array(valid_errors)
    yields_norm = raw_yields / raw_yields[0]
    errors_norm = valid_errs / raw_yields[0]
    plt.figure(figsize=(9, 6))
    plt.errorbar(cumulative_charges, yields_norm, yerr=errors_norm, fmt='o', color='k', ecolor='k', capsize=4, label=r"p$_0$ Normalised Yields")
    x = cumulative_charges
    y = yields_norm
    x_fit = np.linspace(x.min(), x.max(), 100)
    if FORCE_LINEAR_FIT:
        x0 = x[0]
        y0 = y[0]
        weights = 1 / errors_norm**2
        slope_fit = np.sum(weights * (x - x0) * (y - y0)) / np.sum(weights * (x - x0)**2)
        slope_error = np.sqrt(1 / np.sum(weights * (x - x0)**2))
        y_fit = y0 + slope_fit * (x_fit - x0)
        fit_label = f"p$_0$ Fit (slope = {slope_fit:.3g} ± {slope_error:.2g})"
    else:
        coeffs, cov = np.polyfit(x, y, 1, w=1/errors_norm, cov=True)
        slope_fit, intercept_fit = coeffs
        slope_error = np.sqrt(cov[0,0])
        y_fit = slope_fit * x_fit + intercept_fit
        fit_label = f"p$_0$ Fit (slope = {slope_fit:.3g} ± {slope_error:.2g})"
    plt.plot(x_fit, y_fit, 'k--', label=fit_label)
    plt.xlabel("Cumulative Charge (C)", fontsize=16)
    plt.ylabel("Normalized Yield", fontsize=16)
    plt.title(r"Target #7 - 25 μg/cm$^2$ 10B/Cu Scratched", fontsize=20)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper left', fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.show()

if __name__ == "__main__":
    compute_and_plot_yields()
