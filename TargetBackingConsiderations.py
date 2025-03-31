"""
This code is used to produce a combined plot of the non-normalised
initial yield for two targets and their degradation trends
It is used as follows :
    -beam and data file paths for each target of interest are provided in the main function
    with the variables TARGET*_MPA_FILES_PATTERN and TARGET*_BEAM_FILES_PATTERN.
    -Calibration parameters are specified in the slope and intercept variables
    - For files that are to be exclude due to experimental issues
    they can be refernced by their index and excluded for either
    the p1 or p0 plots using and exclude_indices_target*
    - Some files in the early stages of analysis required the data to be loaded
    from a different location and hence these files can be specified
    similarly to the excluded ones using files_for_A_target*.
    - Graphing and changes in the plots have to be handled manually by changing
    titles, labels etc.
    - Plot and fit ranges can be specified and were chosen using techniques
    outlined in the report
    -The option to make the fits have to go through the first data point is chosen
    by setting the variabel force_fit_option to either True or false. Similarly the
    option to make the fit linear or polynomial is available using fit_type_option = 'linear/poly'
"""


import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def gaussian(x, a, mu, sigma):
    return a * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def expected_counts_from_gaussian_fit(energy, B, mu, sigma, bin_width, num_sigma=3):
    mask = (energy >= mu - num_sigma * sigma) & (energy <= mu + num_sigma * sigma)
    masked_energy = energy[mask]
    if len(masked_energy) < 2:
        area = np.sum(gaussian(masked_energy, B, mu, sigma))
    else:
        area = np.trapz(gaussian(masked_energy, B, mu, sigma), masked_energy)
    return area / bin_width

def load_mpa_file_and_time_A(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
        run_time = None
        for line in lines:
            if line.strip().lower().startswith("realtime="):
                try:
                    run_time = float(line.strip().split("=")[1])
                except ValueError:
                    pass
                break
        data_start = None
        for i, line in enumerate(lines):
            if line.startswith("XTITLE") and "1A" in line:
                data_start = i
                break
        if data_start is None:
            print(f"XTITLE 1A not found in {file_path}.")
            return None, run_time
        for i in range(data_start, len(lines)):
            if lines[i].startswith("[DATA]"):
                data_start = i + 1
                break
        data = []
        for line in lines[data_start:]:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            try:
                ch, co = map(int, parts)
                data.append((ch, co))
            except:
                break
        df = pd.DataFrame(data, columns=["Channel", "Counts"])
        return df, run_time
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None

def load_mpa_file_and_time_C(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
        run_time = None
        for line in lines:
            if line.strip().lower().startswith("realtime="):
                try:
                    run_time = float(line.strip().split("=")[1])
                except ValueError:
                    pass
                break
        data_start = None
        for i, line in enumerate(lines):
            if line.startswith("XTITLE") and "1C" in line:
                data_start = i
                break
        if data_start is None:
            print(f"XTITLE 1C not found in {file_path}.")
            return None, run_time
        for i in range(data_start, len(lines)):
            if lines[i].startswith("[DATA]"):
                data_start = i + 1
                break
        data = []
        for line in lines[data_start:]:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            try:
                ch, co = map(int, parts)
                data.append((ch, co))
            except:
                break
        df = pd.DataFrame(data, columns=["Channel", "Counts"])
        return df, run_time
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None

def fit_peak(df, energy_range, slope, intercept):
    df["Energy"] = df["Channel"] * slope + intercept
    region = df[(df["Energy"] >= energy_range[0]) & (df["Energy"] <= energy_range[1])]
    if region.empty:
        return None
    energies = region["Energy"].values
    counts = region["Counts"].values
    if len(energies) < 3:
        return None
    a_guess = counts.max()
    mu_guess = energies[np.argmax(counts)]
    sigma_guess = (energy_range[1] - energy_range[0]) / 6
    try:
        popt, _ = curve_fit(gaussian, energies, counts, p0=[a_guess, mu_guess, sigma_guess])
        return popt
    except:
        return None

def estimate_noise(df, roi_start, roi_end, sigma):
    side_left = df[(df["Energy"] >= roi_start - sigma) & (df["Energy"] < roi_start)]
    side_right = df[(df["Energy"] > roi_end) & (df["Energy"] <= roi_end + sigma)]
    noise_avg = pd.concat([side_left, side_right])["Counts"].mean()
    return int(round(noise_avg)) if not np.isnan(noise_avg) else 0

def process_histogram_p0(df, file_path, slope, intercept, energy_range):
    df["Energy"] = df["Channel"] * slope + intercept
    popt_guess = fit_peak(df, energy_range, slope, intercept)
    if popt_guess is None:
        return None
    a_guess, mu_guess, sig_guess = popt_guess
    roi_start = mu_guess - 3 * sig_guess
    roi_end = mu_guess + 3 * sig_guess
    region = df[(df["Energy"] >= roi_start) & (df["Energy"] <= roi_end)].copy()
    if region.empty:
        return None
    noise_level = estimate_noise(df, roi_start, roi_end, sig_guess)
    region["Counts_Corr"] = region["Counts"] - noise_level
    region.loc[region["Counts_Corr"] < 0, "Counts_Corr"] = 0
    energies_roi = region["Energy"].values
    counts_corr = region["Counts_Corr"].values
    if len(energies_roi) < 3:
        return None
    a_init = counts_corr.max()
    mu_init = energies_roi[np.argmax(counts_corr)]
    sigma_init = (roi_end - roi_start) / 6
    try:
        popt, _ = curve_fit(gaussian, energies_roi, counts_corr, p0=[a_init, mu_init, sigma_init])
    except Exception as e:
        print(f"Error in p0 fitting for {file_path}: {e}")
        return None
    a_fit, mu, sig = popt
    energy_array = np.linspace(roi_start, roi_end, 1000)
    expected_counts_val = expected_counts_from_gaussian_fit(energy_array, a_fit, mu, sig, bin_width=slope, num_sigma=3)
    print(f"[p0] {file_path}: Expected counts = {expected_counts_val:.1f}, sigma = {sig:.3f}")
    return expected_counts_val, mu, sig

def calculate_yield_p0(mpa_files, beam_info_files, slope, intercept, energy_range, exclude_indices=None, files_for_A=None):
    if files_for_A is None:
        files_for_A = []
    cumulative_charge = 0.0
    results = []
    for i, data_file in enumerate(mpa_files):
        if i >= len(beam_info_files):
            continue
        beam_file = beam_info_files[i]
        time_vals, charge_vals = [], []
        with open(beam_file, 'r') as bf:
            for line in bf:
                line = line.strip().rstrip(',')
                if not line:
                    continue
                parts = line.split(',')
                if len(parts) < 2:
                    continue
                try:
                    time_vals.append(float(parts[0]))
                    charge_vals.append(float(parts[1]))
                except:
                    continue
        if len(time_vals) <= 6:
            continue
        time_vals = time_vals[2:-4]
        charge_vals = charge_vals[2:-4]
        sum_charge_C = sum(charge_vals) * 1e-8
        if i in files_for_A:
            df, _ = load_mpa_file_and_time_A(data_file)
        else:
            df, _ = load_mpa_file_and_time_C(data_file)
        if df is None or df.empty:
            continue
        out = process_histogram_p0(df, data_file, slope, intercept, energy_range)
        if not out:
            continue
        expected_counts_val, mu, sig = out
        cumulative_charge += sum_charge_C
        yield_run = expected_counts_val / sum_charge_C if sum_charge_C > 0 else 0
        yield_err = (np.sqrt(expected_counts_val) / sum_charge_C) if expected_counts_val > 0 else 0
        results.append((cumulative_charge, yield_run, yield_err, sig))
        print(f"[p0] Run {i+1}: Expected counts = {expected_counts_val:.1f}, yield = {yield_run}")
    return results

def plot_comparison_p0_non_normalized(results_target1, results_target2, exclude_indices_target1=None, exclude_indices_target2=None, fit_type='linear', force_fit=True):
    xp1, yp1, ye1, _ = zip(*results_target1)
    xp1 = np.array(xp1)
    yp1 = np.array(yp1)
    ye1 = np.array(ye1)
    xp2, yp2, ye2, _ = zip(*results_target2)
    xp2 = np.array(xp2)
    yp2 = np.array(yp2)
    ye2 = np.array(ye2)
    if exclude_indices_target1 is not None:
        include_idx1 = [i for i in range(len(xp1)) if i not in exclude_indices_target1]
    else:
        include_idx1 = list(range(len(xp1)))
    if not include_idx1:
        include_idx1 = list(range(len(xp1)))
    if exclude_indices_target2 is not None:
        include_idx2 = [i for i in range(len(xp2)) if i not in exclude_indices_target2]
    else:
        include_idx2 = list(range(len(xp2)))
    if not include_idx2:
        include_idx2 = list(range(len(xp2)))
    xp1_fit = xp1[include_idx1]
    yp1_fit = yp1[include_idx1]
    ye1_fit = ye1[include_idx1]
    xp2_fit = xp2[include_idx2]
    yp2_fit = yp2[include_idx2]
    ye2_fit = ye2[include_idx2]
    global_x_min = min(xp1.min(), xp2.min())
    global_x_max = max(xp1.max(), xp2.max())
    x_fit_global = np.linspace(global_x_min, global_x_max, 100)
    if fit_type == 'linear':
        if force_fit:
            x0_1 = xp1_fit[0]
            y0_1 = yp1_fit[0]
            slope1 = np.sum((xp1_fit - x0_1) * (yp1_fit - y0_1)) / np.sum((xp1_fit - x0_1)**2)
            residuals1 = yp1_fit - (slope1 * (xp1_fit - x0_1) + y0_1)
            n1 = len(xp1_fit)
            slope_err1 = np.sqrt(np.sum(residuals1**2) / (n1 - 1)) / np.sqrt(np.sum((xp1_fit - x0_1)**2))
            y_fit1 = slope1 * (x_fit_global - x0_1) + y0_1
            x0_2 = xp2_fit[0]
            y0_2 = yp2_fit[0]
            slope2 = np.sum((xp2_fit - x0_2) * (yp2_fit - y0_2)) / np.sum((xp2_fit - x0_2)**2)
            residuals2 = yp2_fit - (slope2 * (xp2_fit - x0_2) + y0_2)
            n2 = len(xp2_fit)
            slope_err2 = np.sqrt(np.sum(residuals2**2) / (n2 - 1)) / np.sqrt(np.sum((xp2_fit - x0_2)**2))
            y_fit2 = slope2 * (x_fit_global - x0_2) + y0_2
        else:
            coeffs1, cov1 = np.polyfit(xp1_fit, yp1_fit, 1, cov=True)
            slope1, intercept1 = coeffs1
            slope_err1 = np.sqrt(cov1[0,0])
            y_fit1 = slope1 * x_fit_global + intercept1
            coeffs2, cov2 = np.polyfit(xp2_fit, yp2_fit, 1, cov=True)
            slope2, intercept2 = coeffs2
            slope_err2 = np.sqrt(cov2[0,0])
            y_fit2 = slope2 * x_fit_global + intercept2
    else:
        x_fit_global = xp1_fit
        y_fit1 = yp1_fit
        y_fit2 = yp2_fit
        slope1 = slope2 = slope_err1 = slope_err2 = np.nan
    hex_color = '#FCD5B5'
    fig, ax = plt.subplots(figsize=(10,6))
    fig.patch.set_facecolor(hex_color)
    ax.set_facecolor(hex_color)
    ax.plot(x_fit_global, y_fit1, 'k--', label="Smooth Ta p$_0$ Degradation Trend ", color='k')
    ax.plot(x_fit_global, y_fit2, 'r--', label="Smooth Cu p$_0$ Degradation Trend", color='r')
    ax.errorbar(xp1_fit[0], yp1_fit[0], yerr=ye1_fit[0], fmt='ko', capsize=4, label="Smooth Ta Initial Yield")
    ax.errorbar(xp2_fit[0], yp2_fit[0], yerr=ye2_fit[0], fmt='ro', capsize=4, label="Smooth Cu Initial Yield")
    ax.set_xlabel("Cumulative Charge (C)", fontsize=21)
    ax.set_ylabel("Yield (Arb. Units)", fontsize=21)
    ax.set_ylim(ymin=1e5)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    legend = ax.legend(loc='lower left', fontsize=14.5, edgecolor='k')
    legend.get_frame().set_facecolor(hex_color)
    ax.set_ylim(bottom=2.2e5)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.grid(True)
    plt.show()

if __name__ == '__main__':
    p0_slope = 0.001847
    p0_intercept = 0.613362
    p0_energy_range = (3.5, 4.5)
    force_fit_option = True  
    fit_type_option = 'linear'
    
    TARGET1_MPA_FILES_PATTERN = r"C:\SH Data Files\Data Files\Target #2\10B_a_targ_test_*.mpa"
    TARGET1_BEAM_FILES_PATTERN = r"C:\SH Data Files\Beam Files\Target #2\10B_a_targ_test*.dat"
    TARGET2_MPA_FILES_PATTERN = r"C:\SH Data Files\Data Files\Target #4\10B_a_targ_test_*.mpa"
    TARGET2_BEAM_FILES_PATTERN = r"C:\SH Data Files\Beam Files\Target #4\10B_a_targ_test*.dat"
    
    mpa_files_p0_target1 = sorted(glob.glob(TARGET1_MPA_FILES_PATTERN))
    beam_files_p0_target1 = sorted(glob.glob(TARGET1_BEAM_FILES_PATTERN))
    files_for_A_p0_target1 = []
    results_p0_target1 = calculate_yield_p0(mpa_files_p0_target1, beam_files_p0_target1,
                                           p0_slope, p0_intercept, p0_energy_range, files_for_A=files_for_A_p0_target1)
    
    mpa_files_p0_target2 = sorted(glob.glob(TARGET2_MPA_FILES_PATTERN))
    beam_files_p0_target2 = sorted(glob.glob(TARGET2_BEAM_FILES_PATTERN))
    files_for_A_p0_target2 = []
    results_p0_target2 = calculate_yield_p0(mpa_files_p0_target2, beam_files_p0_target2,
                                           p0_slope, p0_intercept, p0_energy_range, files_for_A=files_for_A_p0_target2)
    
    exclude_indices_target1 = []
    exclude_indices_target2 = []
    
    plot_comparison_p0_non_normalized(results_p0_target1, results_p0_target2,
                                      exclude_indices_target1=exclude_indices_target1,
                                      exclude_indices_target2=exclude_indices_target2,
                                      fit_type=fit_type_option, force_fit=force_fit_option)
