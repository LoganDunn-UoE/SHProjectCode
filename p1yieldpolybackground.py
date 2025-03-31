"""
This code is used to produce a combined plot of the normalised
degradation trend for the p1 region and the model for the
background and gaussian in that region
It is used as follows :
    -beam and data file paths for each target of interest are provided in the main function
    with the variables MPA_FILES_PATTERN and BEAM_FILES_PATTERN. The beam files pattern input
    is located further up in the code.
    -Calibration parameters are specified in the slope and intercept variables
    - For files that are to be exclude due to experimental issues
    they can be refernced by their index and excluded for
    the p1 plot using and exclude_indices
    - Graphing and changes in the plots have to be handled manually by changing
    titles, labels etc.
    - Plot and fit ranges can be specified and were chosen using techniques
    outlined in the report

"""

import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from math import sqrt, pi

def expected_counts_from_gaussian_fit(energy, B, mu, sigma, bin_width, num_sigma=3):
    mask = (energy >= mu - num_sigma * sigma) & (energy <= mu + num_sigma * sigma)
    masked_energy = energy[mask]
    if len(masked_energy) < 2:
        area = np.sum(gaussian(masked_energy, B, mu, sigma))
    else:
        area = np.trapz(gaussian(masked_energy, B, mu, sigma), masked_energy)
    return area / bin_width

def gaussian(x, a, mu, sigma):
    return a * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def process_histogram(df, file_path, slope, intercept, fit_energy_range, plot_energy_range):
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
        popt, pcov = curve_fit(
            model_poly_sq_gauss,
            energy_fit,
            counts_fit,
            p0=[a0_init, a1_init, a2_init, area_init, mu_init, sigma_init],
            sigma=np.sqrt(counts_fit + 1),
            bounds=(
                [-np.inf, -np.inf, -np.inf, 0, fit_energy_range[0], 1e-9],
                [ np.inf,  np.inf,  np.inf, np.inf, fit_energy_range[1], np.inf]
            )
        )
        a0_fit, a1_fit, a2_fit, area_fit, mu_fit, sigma_fit = popt
        fit_vals = model_poly_sq_gauss(energy_fit, *popt)
        errors = np.sqrt(counts_fit + 1)
        residuals = counts_fit - fit_vals
        weighted_residuals = residuals / errors
        chi2 = np.sum(weighted_residuals**2)
        ndf = len(counts_fit) - len(popt)
        if ndf < 1: ndf = 1
        red_chi2 = chi2 / ndf
        print(f"\nFor {file_path}:")
        print(f"  Chi2 = {chi2:.3f}, NDF = {ndf}, Reduced chi2 = {red_chi2:.3f}")
        df_plot = df[(df["Energy"] >= plot_energy_range[0]) & (df["Energy"] <= plot_energy_range[1])].copy()
        energy_plot = df_plot["Energy"].values
        counts_plot = df_plot["Counts"].values
        raw_poly = (a0_fit + a1_fit * energy_plot + a2_fit * (energy_plot**2))**2
        gauss_norm = (area_fit / (sqrt(2.0 * pi) * sigma_fit)) * np.exp(-((energy_plot - mu_fit)**2) / (2 * sigma_fit**2))
        sort_idx = np.argsort(energy_plot)
        energy_plot = energy_plot[sort_idx]
        counts_plot = counts_plot[sort_idx]
        raw_poly = raw_poly[sort_idx]
        gauss_norm = gauss_norm[sort_idx]
        min_idx = np.argmin(raw_poly)
        min_val = raw_poly[min_idx]
        for j in range(min_idx, len(raw_poly)):
            raw_poly[j] = min_val
        full_fit = raw_poly + gauss_norm
        B = area_fit / (sqrt(2.0 * pi) * sigma_fit)
        bg_removed_gauss = gaussian(energy_plot, B, mu_fit, sigma_fit)
        yield_counts = expected_counts_from_gaussian_fit(energy_plot, B, mu_fit, sigma_fit, bin_width=slope, num_sigma=3)
        print(f"  Expected counts (±3σ) from fitted Gaussian = {yield_counts:.1f} counts")
        left_bound = mu_fit - 3 * sigma_fit
        right_bound = mu_fit + 3 * sigma_fit
        plt.figure(figsize=(10, 5))
        bin_width_val = max(slope, 1e-9)
        bins_approx = max(1, int(round((plot_energy_range[1] - plot_energy_range[0]) / bin_width_val)))
        plt.hist(energy_plot, bins=bins_approx, weights=counts_plot, alpha=0.3  )
        plt.plot(energy_plot, counts_plot, 'b-', alpha=0.6)
        plt.plot(energy_plot, full_fit, 'r-', label='Combined Polynomial + Gaussian Model')
        plt.plot(energy_plot, raw_poly, 'g--', label='Polynomial Background')
        plt.plot(energy_plot, gauss_norm, 'm-.', label=r'Gaussian p$_1$ peak')
        plt.hist(energy_plot, bins=bins_approx, weights=bg_removed_gauss, histtype='bar', color='c')
        
        plt.xlabel("Energy (MeV)", fontsize = 14)
        plt.ylabel("Counts", fontsize = 14)
        plt.title(f"Target #4 (15 μg/cm$^2$ 10B/Cu Smooth) p$_1$ Region Model -  Red.χ2={red_chi2:.2f}", fontsize = 14)
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.legend(fontsize =12)
        plt.show()
        partial_chi2 = (residuals / errors)**2
        fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        ax[0].errorbar(energy_fit, residuals, yerr=errors, fmt='o', capsize=3)
        ax[0].axhline(0, color='red', linestyle='--')
        ax[0].set_ylabel("Residuals")
        ax[0].set_title("Residuals vs. Energy")
        """
        ax[1].scatter(energy_fit, partial_chi2, marker='o')
        ax[1].set_xlabel("Energy (MeV)")
        ax[1].set_ylabel("Partial χ2")
        ax[1].set_title("Per-Point χ2 Contributions")
        """
        plt.tight_layout()
        plt.show()
        bg_subtracted_data = counts_plot - raw_poly
        plt.figure(figsize=(10, 5))
        plt.hist(energy_plot, bins=bins_approx, weights=bg_subtracted_data, alpha=0.7, label='Data (BG-subtracted)')
        plt.plot(energy_plot, bg_removed_gauss, 'c--',color ='black', label='Fitted BG-Removed Gaussian')
        plt.xlabel("Energy (MeV)")
        plt.ylabel("Counts (BG Subtracted)")
        plt.title("Background-Subtracted Data vs. Fitted Gaussian")
        plt.legend()
        plt.show()
        return yield_counts, mu_fit, sigma_fit
    except Exception as e:
        print(f"Error fitting {file_path}: {e}")
        return None

def calculate_yield_plot(mpa_files, fit_energy_range, plot_energy_range, slope, intercept, beam_currents, exclude_yield_indices=None):
    cumulative_charge = 0.0
    results = []
    beam_info_files = sorted(glob.glob(r"C:\SH Data Files\Beam Files\Target #4\10B_a_targ_test*.dat"))
    for i, data_file in enumerate(mpa_files):
        if i >= len(beam_info_files):
            continue
        beam_file = beam_info_files[i]
        t_vals = []
        c_vals = []
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
        df, _ = load_mpa_file_and_time(data_file)
        if df is None or df.empty:
            continue
        out = process_histogram(df, data_file, slope, intercept, fit_energy_range, plot_energy_range)
        if not out:
            continue
        expected_counts_val, mu_fit, sigma_fit = out
        cumulative_charge += sum_C
        yield_run = expected_counts_val / sum_C if sum_C > 0 else 0
        yield_err = (np.sqrt(expected_counts_val) / sum_C) if expected_counts_val > 0 else 0
        results.append((cumulative_charge, yield_run, yield_err, sigma_fit))
        print(f"[p1] Run {i+1}: Expected counts = {expected_counts_val:.1f}, sigma = {sigma_fit:.3f}")
    if results:
        xp, yp, ye, sp = zip(*results)
        xp = np.array(xp)
        yp = np.array(yp)
        ye = np.array(ye)
        if exclude_yield_indices is not None:
            include_idx = sorted([i for i in range(len(xp)) if i not in exclude_yield_indices])
        else:
            include_idx = list(range(len(xp)))
        base_yield = yp[include_idx[0]]
        yn = yp / base_yield
        yen = ye / base_yield
        xp_incl = xp[include_idx]
        yn_incl = yn[include_idx]
        yen_incl = yen[include_idx]
        plt.figure()
        plt.scatter(xp, yn)
        plt.errorbar(xp_incl, yn_incl, yerr=yen_incl, fmt='o', capsize=4, color='b')
        plt.xlabel("Cumulative Charge (C)")
        plt.ylabel("Normalized Yield")
        plt.title("")
        #plt.xlim(xmin =)
        #plt.xlim(xmax =)
        #plt.ylim(ymin =)
        #plt.ylim(ymax= )
        
        plt.grid(True)
        x0 = xp_incl[0]
        y0 = yn_incl[0]
        slope_fit = np.sum((xp_incl - x0) * (yn_incl - y0)) / np.sum((xp_incl - x0)**2)
        x_fit = np.linspace(xp.min(), xp.max(), 100)
        y_fit = slope_fit * x_fit + (y0 - slope_fit * x0)
        plt.plot(x_fit, y_fit, label=f'Forced Fit (m={slope_fit:.3g})', color='green')
        plt.legend()
        plt.show()
        plt.figure()
        """
        plt.scatter(xp, sp, color='r')
        plt.plot(xp, sp, '--', color='r', label='Sigma Trend')
        plt.xlabel("Cumulative Charge (C)")
        plt.ylabel("Sigma")
        plt.title("p1: Sigma vs. Cumulative Charge")
        plt.grid(True)
        plt.legend()
        plt.show()
        """
    return results

def load_mpa_file_and_time(file_path):
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

def histogram_energy(exclude_yield_indices_p1=None):
    mpa_files_pattern = r"C:\SH Data Files\Data Files\Target #4\10B_a_targ_test_*.mpa"
    mpa_files = glob.glob(mpa_files_pattern)
    slope = 0.001847
    intercept = 0.613362
    fit_range = (0.64, 0.99)
    plot_range = (0.64, 1.1)
    if not mpa_files:
        print("No p1 files found!")
        results = []
    else:
        results = calculate_yield_plot(mpa_files, fit_range, plot_range, slope, intercept, beam_currents=[],
                                       exclude_yield_indices=exclude_yield_indices_p1)
    return results

if __name__ == "__main__":
    histogram_energy(exclude_yield_indices_p1=[])
