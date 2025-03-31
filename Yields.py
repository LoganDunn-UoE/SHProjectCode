"""
This code is used to produce normalised yield plots for the p0, p1 
regions aswell as combining them in a final plot
It is used as follows :
    -beam and data file paths are provided in the main function
    with the variables MPA_FILES_PATTERN and BEAM_FILES_PATTERN.
    -Calibration parameters are specified in the slope and intercept variables
    - For files that are to be exclude due to experimental issues
    they can be refernced by their index and excluded for either
    the p1 or p0 plots using exclude_indices_p1 and exclude_indices_p0
    - Some files in the early stages of analysis required the data to be loaded
    from a different location and hence these files can be specified
    similarly to the excluded ones using files_for_A_p0.
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
from math import sqrt, pi

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

def process_histogram_p1(df, file_path, slope, intercept, fit_energy_range, plot_energy_range):
    df["Energy"] = df["Channel"] * slope + intercept
    df_fit = df[(df["Energy"] >= fit_energy_range[0]) & (df["Energy"] <= fit_energy_range[1])].copy()
    energy_fit = df_fit["Energy"].values
    counts_fit = df_fit["Counts"].values
    a0_init = np.sqrt(np.median(counts_fit)) if np.median(counts_fit) > 0 else 1.0
    a1_init = 0.0
    a2_init = 0.0
    area_init = counts_fit.max() * (fit_energy_range[1] - fit_energy_range[0]) / 2.0
    mu_init = 0.82
    sigma_init = 0.03
    def model_poly_sq_gauss(E, a0, a1, a2, area, mu, sigma):
        poly_bg = (a0 + a1 * E + a2 * (E**2))**2
        gauss_part = (area / (sqrt(2.0 * pi) * sigma)) * np.exp(-((E - mu)**2) / (2 * sigma**2))
        return poly_bg + gauss_part
    try:
        popt, _ = curve_fit(
            model_poly_sq_gauss,
            energy_fit,
            counts_fit,
            p0=[a0_init, a1_init, a2_init, area_init, mu_init, sigma_init],
            sigma=np.sqrt(counts_fit + 1),
            bounds=([-np.inf, -np.inf, -np.inf, 0, fit_energy_range[0], 1e-9],
                    [np.inf, np.inf, np.inf, np.inf, fit_energy_range[1], np.inf])
        )
        a0_fit, a1_fit, a2_fit, area_fit, mu_fit, sigma_fit = popt
        B = area_fit / (sqrt(2.0 * pi) * sigma_fit)
        energy_plot = np.linspace(plot_energy_range[0], plot_energy_range[1], 1000)
        yield_counts = expected_counts_from_gaussian_fit(energy_plot, B, mu_fit, sigma_fit, bin_width=slope, num_sigma=3)
        print(f"[p1] {file_path}: Expected counts = {yield_counts:.1f}, mu = {mu_fit:.3f}, sigma = {sigma_fit:.3f}")
        return yield_counts, mu_fit, sigma_fit
    except Exception as e:
        print(f"Error in p1 fitting for {file_path}: {e}")
        return None

def calculate_yield_p1(mpa_files, beam_info_files, slope, intercept, fit_energy_range, plot_energy_range, exclude_indices=None):
    cumulative_charge = 0.0
    results = []
    for i, data_file in enumerate(mpa_files):
        if i >= len(beam_info_files):
            continue
        beam_file = beam_info_files[i]
        t_vals, c_vals = [], []
        with open(beam_file, 'r') as bf:
            for line in bf:
                line = line.strip().rstrip(',')
                if not line:
                    continue
                parts = line.split(',')
                if len(parts) < 2:
                    continue
                try:
                    t_vals.append(float(parts[0]))
                    c_vals.append(float(parts[1]))
                except:
                    continue
        if len(t_vals) <= 6:
            continue
        t_vals = t_vals[2:-4]
        c_vals = c_vals[2:-4]
        sum_C = sum(c_vals) * 1e-8
        df, _ = load_mpa_file_and_time_C(data_file)
        if df is None or df.empty:
            continue
        out = process_histogram_p1(df, data_file, slope, intercept, fit_energy_range, plot_energy_range)
        if not out:
            continue
        expected_counts, mu_fit, sigma_fit = out
        cumulative_charge += sum_C
        yield_run = expected_counts / sum_C if sum_C > 0 else 0
        yield_err = (np.sqrt(expected_counts) / sum_C) if expected_counts > 0 else 0
        results.append((cumulative_charge, yield_run, yield_err, sigma_fit))
        print(f"[p1] Run {i+1}: Expected counts = {expected_counts:.1f}, sigma = {sigma_fit:.3f}")
    return results

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

def plot_yield(results, title, color='b', label='Data', exclude_indices=None, force_fit=True, open_marker=False, fit_type='linear', poly_degree=2):
    if not results:
        print("No results to plot for", title)
        return
    xp, yp, ye, _ = zip(*results)
    xp = np.array(xp)
    yp = np.array(yp)
    ye = np.array(ye)
    if exclude_indices is not None:
        include_idx = [i for i in range(len(xp)) if i not in exclude_indices]
    else:
        include_idx = list(range(len(xp)))
    if not include_idx:
        include_idx = list(range(len(xp)))
    base_yield = yp[include_idx[0]]
    normalized_yield = yp / base_yield
    normalized_err = ye / base_yield
    hex_color = '#FCD5B5'
    fig, ax = plt.subplots(figsize=(10,6))
    fig.patch.set_facecolor(hex_color)
    ax.set_facecolor(hex_color)
    ax.scatter(xp, normalized_yield, color=color)
    if open_marker:
        ax.errorbar(xp[include_idx], normalized_yield[include_idx], yerr=normalized_err[include_idx],
                    fmt='o', capsize=4, color=color, markerfacecolor='none', label=label + " (included)")
    else:
        ax.errorbar(xp[include_idx], normalized_yield[include_idx], yerr=normalized_err[include_idx],
                    fmt='o', capsize=4, color=color, label=r"p$_0$ Normalised Yields")
    ax.set_xlabel("Cumulative Charge (C)", fontsize=21)
    ax.set_ylabel("Normalized Yield", fontsize=21)
    x_fit_points = xp[include_idx]
    y_fit_points = normalized_yield[include_idx]
    x_fit = np.linspace(xp.min(), xp.max(), 100)
    if fit_type == 'linear':
        if force_fit:
            x0 = x_fit_points[0]
            y0 = y_fit_points[0]
            slope_fit = np.sum((x_fit_points - x0) * (y_fit_points - y0)) / np.sum((x_fit_points - x0)**2)
            y_fit = slope_fit * (x_fit - x0) + y0
            residuals = y_fit_points - (slope_fit * (x_fit_points - x0) + y0)
            n = len(x_fit_points)
            sigma_m = np.sqrt(np.sum(residuals**2) / (n - 1)) / np.sqrt(np.sum((x_fit_points - x0)**2))
            ax.plot(x_fit, y_fit, 'k--', label=r"p$_0$" + f' Fit (slope={slope_fit:.3g}±{sigma_m:.1g})', color=color)
        else:
            coeffs, cov = np.polyfit(x_fit_points, y_fit_points, 1, cov=True)
            slope_fit, intercept_fit = coeffs
            slope_err = sqrt(cov[0,0])
            y_fit = slope_fit * x_fit + intercept_fit
            ax.plot(x_fit, y_fit, 'k--', label=f'Fit (slope={slope_fit:.3g}±{slope_err:.1g})', color=color)
    elif fit_type == 'poly':
        if force_fit:
            x0 = x_fit_points[0]
            y0 = y_fit_points[0]
            if poly_degree == 2:
                X = x_fit_points - x0
                Y = y_fit_points - y0
                A = np.sum((X**2) * Y) / np.sum((X**2)**2)
                y_fit = y0 + A * (x_fit - x0)**2
            else:
                X = x_fit_points - x0
                Y = y_fit_points - y0
                D = np.vstack([X**i for i in range(1, poly_degree+1)]).T
                coeffs = np.linalg.lstsq(D, Y, rcond=None)[0]
                X_fit = x_fit - x0
                y_fit = y0 + np.sum([coeffs[i] * (X_fit**(i+1)) for i in range(poly_degree)], axis=0)
            ax.plot(x_fit, y_fit, 'k--', label=r'p$_0$ Fit', color=color)
        else:
            coeffs = np.polyfit(x_fit_points, y_fit_points, poly_degree)
            y_fit = np.polyval(coeffs, x_fit)
            ax.plot(x_fit, y_fit, 'k--', label=r'p$_0$ Fit', color=color)
    ax.set_ylim(bottom=0.9)
    legend = ax.legend(loc='upper right', fontsize=18.5, edgecolor='k')
    legend.get_frame().set_facecolor(hex_color)
    ax.tick_params(axis='both', which='major', labelsize=21)
    ax.grid(True)
    plt.show()

def plot_combined_yield(results_p0, results_p1, exclude_indices_p0=None, exclude_indices_p1=None, force_fit=True, fit_type='linear', poly_degree=2):
    if not results_p0 or not results_p1:
        print("Insufficient data for combined plot.")
        return
    xp0, yp0, ye0, _ = zip(*results_p0)
    xp1, yp1, ye1, _ = zip(*results_p1)
    xp0 = np.array(xp0)
    yp0 = np.array(yp0)
    ye0 = np.array(ye0)
    xp1 = np.array(xp1)
    yp1 = np.array(yp1)
    ye1 = np.array(ye1)
    if exclude_indices_p0 is not None:
        include_idx0 = [i for i in range(len(xp0)) if i not in exclude_indices_p0]
    else:
        include_idx0 = list(range(len(xp0)))
    if not include_idx0:
        include_idx0 = list(range(len(xp0)))
    base0 = yp0[include_idx0[0]]
    norm0 = yp0 / base0
    if exclude_indices_p1 is not None:
        include_idx1 = [i for i in range(len(xp1)) if i not in exclude_indices_p1]
    else:
        include_idx1 = list(range(len(xp1)))
    if not include_idx1:
        include_idx1 = list(range(len(xp1)))
    base1 = yp1[include_idx1[0]]
    norm1 = yp1 / base1
    hex_color = '#FCD5B5'
    fig, ax = plt.subplots(figsize=(10,6))
    fig.patch.set_facecolor(hex_color)
    ax.set_facecolor(hex_color)
    ax.errorbar(xp0, norm0, yerr=ye0/base0, fmt='o', capsize=4, color='k', label=r"p$_0$ Normalised Yields")
    ax.errorbar(xp1, norm1, yerr=ye1/base1, fmt='o', capsize=4, color='r', markerfacecolor='none', label=r"p$_1$ Normalised Yields")
    x_fit_points0 = xp0[include_idx0]
    y_fit_points0 = norm0[include_idx0]
    x_fit0 = np.linspace(xp0.min(), xp0.max(), 100)
    if fit_type == 'linear':
        if force_fit:
            x0_0 = x_fit_points0[0]
            y0_0 = y_fit_points0[0]
            slope_fit0 = np.sum((x_fit_points0 - x0_0) * (y_fit_points0 - y0_0)) / np.sum((x_fit_points0 - x0_0)**2)
            residuals0 = y_fit_points0 - (slope_fit0 * (x_fit_points0 - x0_0) + y0_0)
            n0 = len(x_fit_points0)
            sigma_m0 = np.sqrt(np.sum(residuals0**2) / (n0 - 1)) / np.sqrt(np.sum((x_fit_points0 - x0_0)**2))
            y_fit0 = slope_fit0 * (x_fit0 - x0_0) + y0_0
            ax.plot(x_fit0, y_fit0, 'k--', label=f'p$_0$ Fit (slope={slope_fit0:.3g}±{sigma_m0:.1g})')
        else:
            coeffs0, cov0 = np.polyfit(x_fit_points0, y_fit_points0, 1, cov=True)
            slope_fit0, intercept_fit0 = coeffs0
            slope_err0 = sqrt(cov0[0,0])
            y_fit0 = slope_fit0 * x_fit0 + intercept_fit0
            ax.plot(x_fit0, y_fit0, 'k--', label=f'p$_0$ Fit (slope={slope_fit0:.3g}±{slope_err0:.1g})')
    elif fit_type == 'poly':
        if force_fit:
            x0_0 = x_fit_points0[0]
            y0_0 = y_fit_points0[0]
            if poly_degree == 2:
                X0 = x_fit_points0 - x0_0
                Y0 = y_fit_points0 - y0_0
                A0 = np.sum((X0**2) * Y0) / np.sum((X0**2)**2)
                y_fit0 = y0_0 + A0 * (x_fit0 - x0_0)**2
            else:
                X0 = x_fit_points0 - x0_0
                Y0 = y_fit_points0 - y0_0
                D0 = np.vstack([X0**i for i in range(1, poly_degree+1)]).T
                coeffs0 = np.linalg.lstsq(D0, Y0, rcond=None)[0]
                X_fit0 = x_fit0 - x0_0
                y_fit0 = y0_0 + np.sum([coeffs0[i] * (X_fit0**(i+1)) for i in range(poly_degree)], axis=0)
            ax.plot(x_fit0, y_fit0, 'k--', label=r'p$_0$ Fit')
        else:
            coeffs0 = np.polyfit(x_fit_points0, y_fit_points0, poly_degree)
            y_fit0 = np.polyval(coeffs0, x_fit0)
            ax.plot(x_fit0, y_fit0, 'k--', label=r'p$_0$ Fit')
    x_fit_points1 = xp1[include_idx1]
    y_fit_points1 = norm1[include_idx1]
    x_fit1 = np.linspace(xp1.min(), xp1.max(), 100)
    if fit_type == 'linear':
        if force_fit:
            x0_1 = x_fit_points1[0]
            y0_1 = y_fit_points1[0]
            slope_fit1 = np.sum((x_fit_points1 - x0_1) * (y_fit_points1 - y0_1)) / np.sum((x_fit_points1 - x0_1)**2)
            residuals1 = y_fit_points1 - (slope_fit1 * (x_fit_points1 - x0_1) + y0_1)
            n1 = len(x_fit_points1)
            sigma_m1 = np.sqrt(np.sum(residuals1**2) / (n1 - 1)) / np.sqrt(np.sum((x_fit_points1 - x0_1)**2))
            y_fit1 = slope_fit1 * (x_fit1 - x0_1) + y0_1
            ax.plot(x_fit1, y_fit1, 'r--', label=f'p$_1$ Fit (slope={slope_fit1:.3g}±{sigma_m1:.1g})', color='r')
        else:
            coeffs1, cov1 = np.polyfit(x_fit_points1, y_fit_points1, 1, cov=True)
            slope_fit1, intercept_fit1 = coeffs1
            slope_err1 = sqrt(cov1[0,0])
            y_fit1 = slope_fit1 * x_fit1 + intercept_fit1
            ax.plot(x_fit1, y_fit1, 'r--', label=f'p$_1$ Fit (slope={slope_fit1:.3g}±{slope_err1:.1g})', color='r')
    elif fit_type == 'poly':
        if force_fit:
            x0_1 = x_fit_points1[0]
            y0_1 = y_fit_points1[0]
            if poly_degree == 2:
                X1 = x_fit_points1 - x0_1
                Y1 = y_fit_points1 - y0_1
                A1 = np.sum((X1**2) * Y1) / np.sum((X1**2)**2)
                y_fit1 = y0_1 + A1 * (x_fit1 - x0_1)**2
            else:
                X1 = x_fit_points1 - x0_1
                Y1 = y_fit_points1 - y0_1
                D1 = np.vstack([X1**i for i in range(1, poly_degree+1)]).T
                coeffs1 = np.linalg.lstsq(D1, Y1, rcond=None)[0]
                X_fit1 = x_fit1 - x0_1
                y_fit1 = y0_1 + np.sum([coeffs1[i] * (X_fit1**(i+1)) for i in range(poly_degree)], axis=0)
            ax.plot(x_fit1, y_fit1, 'r--', label=f'p1 Fit ', color='r')
        else:
            coeffs1 = np.polyfit(x_fit_points1, y_fit_points1, poly_degree)
            y_fit1 = np.polyval(coeffs1, x_fit1)
            ax.plot(x_fit1, y_fit1, 'r--', label=f'p1 Fit ', color='r')
    ax.set_xlabel("Cumulative Charge (C)", fontsize=20)
    ax.set_ylabel("Normalized Yield", fontsize=20)
    ax.set_xlim(right=1)
    ax.set_ylim(bottom=0.85)
    ax.grid(True)
    legend = ax.legend(loc='lower left', fontsize=13, edgecolor='k')
    legend.get_frame().set_facecolor(hex_color)
    ax.tick_params(axis='both', which='major', labelsize=16)
    plt.show()

if __name__ == '__main__':
    p1_slope = 0.001847
    p1_intercept = 0.613362
    p1_fit_range = (0.64, 0.99)
    p1_plot_range = (0.64, 1.1)
    p0_slope = 0.001847
    p0_intercept = 0.613362
    p0_energy_range = (3.5, 4.5)
    
    MPA_FILES_PATTERN = r"C:\SH Data Files\Data Files\Target #2\10B_a_targ_test_*.mpa"
    BEAM_FILES_PATTERN = r"C:\SH Data Files\Beam Files\Target #2\10B_a_targ_test*.dat"
    
    exclude_indices_p1 = [1,2]
    
    exclude_indices_p0 = [1,2]
    force_fit_option = True
    fit_type_option = 'linear'
    poly_degree_option = 2
    mpa_files_all = sorted(glob.glob(MPA_FILES_PATTERN))
    beam_files_all = sorted(glob.glob(BEAM_FILES_PATTERN))
    mpa_files_p1 = mpa_files_all
    beam_files_p1 = beam_files_all
    mpa_files_p0 = mpa_files_all
    beam_files_p0 = beam_files_all
    results_p1 = calculate_yield_p1(mpa_files_p1, beam_files_p1, p1_slope, p1_intercept, p1_fit_range, p1_plot_range)
    files_for_A_p0 = [0,1,2,3,4,5,6]
    results_p0 = calculate_yield_p0(mpa_files_p0, beam_files_p0, p0_slope, p0_intercept, p0_energy_range, files_for_A=files_for_A_p0)
    plot_yield(results_p0, "p0 Normalized Yield vs. Cumulative Charge", color='k', label='p0',
               exclude_indices=exclude_indices_p0, force_fit=force_fit_option, fit_type=fit_type_option, poly_degree=poly_degree_option)
    plot_yield(results_p1, "p1 Normalized Yield vs. Cumulative Charge", color='r', label='p1',
               exclude_indices=exclude_indices_p1, force_fit=force_fit_option, open_marker=True, fit_type=fit_type_option, poly_degree=poly_degree_option)
    plot_combined_yield(results_p0, results_p1, exclude_indices_p0=exclude_indices_p0,
                        exclude_indices_p1=exclude_indices_p1, force_fit=force_fit_option, fit_type=fit_type_option, poly_degree=poly_degree_option)
