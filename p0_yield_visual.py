"""
This code is used to produce a combined plot of the normalised
degradation trend for the p0  region and the model for the
background and gaussian in that region
It is used as follows :
    -beam and data file paths for each target of interest are provided in the main function
    with the variables MPA_FILES_PATTERN and BEAM_FILES_PATTERN.
    -Calibration parameters are specified in the slope and intercept variables
    - Graphing and changes in the plots have to be handled manually by changing
    titles, labels etc.
    -The energy ranges for fitting are to be specified using p0_energy_range.
    - Plot and fit ranges can be specified and were chosen using techniques
    outlined in the report

"""


import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def expected_counts_from_gaussian_fit(energy, B, mu, sigma, bin_width, num_sigma=3):
    mask = (energy >= mu - num_sigma * sigma) & (energy <= mu + num_sigma * sigma)
    masked_energy = energy[mask]
    if len(masked_energy) < 2:
        area = np.sum(gaussian(masked_energy, B, mu, sigma))
    else:
        area = np.trapz(gaussian(masked_energy, B, mu, sigma), masked_energy)
    return area / bin_width

def raw_counts_in_region(region, mu, sigma):
    mask = (region["Energy"] >= mu - 3 * sigma) & (region["Energy"] <= mu + 3 * sigma)
    return region.loc[mask, "Counts_Corr"].sum()

def gaussian(x, a, mu, sigma):
    return a * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def fit_peak(df, energy_range, slope, intercept):
    df["Energy"] = slope * df["Channel"] + intercept
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

def load_mpa_file_and_time(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
        run_time = None
        for line in lines:
            line_stripped = line.strip().lower()
            if line_stripped.startswith("realtime="):
                parts = line_stripped.split("=")
                if len(parts) == 2:
                    try:
                        run_time = float(parts[1])
                    except ValueError:
                        pass
                break
        target_xtitle = "1C"
        data_start = None
        for i, line in enumerate(lines):
            if line.startswith("XTITLE") and target_xtitle in line:
                data_start = i
                break
        if data_start is None:
            print(f"XTITLE {target_xtitle} not found in {file_path}.")
            return None, run_time
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
                ch, co = map(int, line.split())
                data.append((ch, co))
            except ValueError:
                break
        df = pd.DataFrame(data, columns=["Channel", "Counts"])
        return df, run_time
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None

def estimate_noise(df, roi_start, roi_end, sigma):
    side_left = df[(df["Energy"] >= roi_start - sigma) & (df["Energy"] < roi_start)]
    side_right = df[(df["Energy"] > roi_end) & (df["Energy"] <= roi_end + sigma)]
    noise_avg = pd.concat([side_left, side_right])["Counts"].mean()
    return int(round(noise_avg)) if not np.isnan(noise_avg) else 0

def calculate_yield_plot_p0(mpa_files, energy_range, slope, intercept, beam_currents, yield_method="expected"):
    cumulative_charge = 0.0
    results_p0 = []
    beam_variance_list_p0 = []
    beam_info_files = sorted(glob.glob(beam_info_files_pattern))
    for i, data_file in enumerate(mpa_files):
        if i >= len(beam_info_files):
            print(f"No matching beam info for {data_file} (index {i})")
            continue
        beam_file = beam_info_files[i]
        time_vals = []
        charge_vals = []
        with open(beam_file, 'r') as bf:
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
        if len(time_vals) <= 6:
            print(f"Skipping beam file: not enough lines => {beam_file}")
            continue
        time_vals = time_vals[2:-4]
        charge_vals = charge_vals[2:-4]
        sum_charge_10neg8 = sum(charge_vals)
        sum_charge_C = sum_charge_10neg8 * 1e-8
        var_ = np.var(charge_vals, ddof=1)
        std_ = np.std(charge_vals, ddof=1)
        beam_variance_list_p0.append((i + 1, var_, std_))
        df, _ = load_mpa_file_and_time(data_file)
        if df is None or df.empty:
            print(f"No data => {data_file}")
            continue
        df["Energy"] = df["Channel"] * slope + intercept
        popt_guess = fit_peak(df, energy_range, slope, intercept)
        if popt_guess is None:
            print(f"No valid peak guess => {data_file}")
            continue
        a_guess, mu_guess, sig_guess = popt_guess
        roi_start = mu_guess - 3 * sig_guess
        roi_end = mu_guess + 3 * sig_guess
        region = df[(df["Energy"] >= roi_start) & (df["Energy"] <= roi_end)].copy()
        if region.empty:
            print(f"No data in ROI for {data_file}")
            continue
        noise_level = estimate_noise(df, roi_start, roi_end, sig_guess)
        region["Counts_Corr"] = region["Counts"] - noise_level
        region.loc[region["Counts_Corr"] < 0, "Counts_Corr"] = 0
        energies_roi = region["Energy"].values
        counts_corr = region["Counts_Corr"].values
        if len(energies_roi) < 3:
            print(f"Not enough points in ROI after background subtraction for {data_file}")
            continue
        a_init = counts_corr.max()
        mu_init = energies_roi[np.argmax(counts_corr)]
        sigma_init = (roi_end - roi_start) / 6
        try:
            popt, _ = curve_fit(gaussian, energies_roi, counts_corr, p0=[a_init, mu_init, sigma_init])
        except Exception as e:
            print(f"Gaussian fit failed on background-subtracted data for {data_file}: {e}")
            continue
        a_fit, mu, sig = popt
        if yield_method == "expected":
            energy_array = np.linspace(roi_start, roi_end, 1000)
            expected_counts_val = int(round(expected_counts_from_gaussian_fit(energy_array, a_fit, mu, sig, bin_width=slope, num_sigma=3)))
        elif yield_method == "raw":
            expected_counts_val = int(round(raw_counts_in_region(region, mu, sig)))
        else:
            energy_array = np.linspace(roi_start, roi_end, 1000)
            expected_counts_val = int(round(expected_counts_from_gaussian_fit(energy_array, a_fit, mu, sig, bin_width=slope, num_sigma=3)))
        print(expected_counts_val)
        cumulative_charge += sum_charge_C
        yield_run = (expected_counts_val / sum_charge_C) if sum_charge_C > 0 else 0
        yield_err = (np.sqrt(expected_counts_val) / sum_charge_C) if expected_counts_val > 0 else 0
        results_p0.append((cumulative_charge, yield_run, yield_err, sig))
        print(f"p0 Run#{i+1}: {data_file}")
        print(f"  beam= {beam_file}, sum= {sum_charge_10neg8:.2f} => {sum_charge_C:.3e} C")
        print(f"  Expected counts (±3σ) from fitted (background-removed) Gaussian = {expected_counts_val:.1f}, sigma= {sig:.3f}")
        plt.figure()
        plt.bar(region["Energy"], region["Counts_Corr"], width=0.002, alpha=0.5, color='gray', label='Background-Removed')
        x_ = np.linspace(roi_start, roi_end, 300)
        plt.plot(x_, gaussian(x_, a_fit, mu, sig), 'r--', label=f'Gauss(μ={mu:.3f}, σ={sig:.3f})')
        plt.xlabel("Energy (MeV)")
        plt.ylabel("Counts")
        plt.title(f"p0: Fitted Gaussian (±3σ region) => {data_file}")
        plt.xlim(roi_start, roi_end)
        plt.legend()
        plt.grid(True)
        plt.show()
        plt.figure()
        plt.plot(time_vals, charge_vals, 'o-')
        plt.title(f"p0 beam => {beam_file}")
        plt.xlabel("Time (arb)")
        plt.ylabel("Charge (×10^-8 C)")
        plt.grid(True)
        plt.show()
    if results_p0:
        xp0, yp0, ye0, sp0 = zip(*results_p0)
        base = yp0[0] if yp0 else 1
        ynp0 = [y / base for y in yp0]
        yenp0 = [e / base for e in ye0]
        plt.figure()
        plt.errorbar(xp0, ynp0, yerr=yenp0, fmt='o', capsize=4, color='b', label='Data')
        plt.plot(xp0, ynp0, '--', color='r', label='Trend')
        plt.xlabel("Cumulative Charge (C)")
        plt.ylabel("Normalized Yield")
        plt.title("p0: Normalized Yield vs. Charge")
        plt.grid(True)
        plt.legend()
        xp0_arr = np.array(xp0)
        ynp0_arr = np.array(ynp0)
        x0 = xp0_arr[0]
        y0 = ynp0_arr[0]
        slope_p0 = np.sum((xp0_arr - x0) * (ynp0_arr - y0)) / np.sum((xp0_arr - x0) ** 2)
        print(f"[p0 Forced Fit] slope={slope_p0:.4g}, passing through ({x0:.4g}, {y0:.4g})")
        x_fit = np.linspace(x0, xp0_arr.max(), 100)
        y_fit = slope_p0 * (x_fit - x0) + y0
        plt.plot(x_fit, y_fit, label=f'Forced Linear Fit (m={slope_p0:.3g})', color='green')
        plt.legend()
        plt.show()
        plt.figure()
        plt.scatter(xp0, sp0, color='r')
        plt.plot(xp0, sp0, '--', color='r', label='Trend')
        plt.xlabel("Cumulative Charge (C)")
        plt.ylabel("Sigma")
        plt.title("p0: Sigma vs. Charge")
        plt.grid(True)
        plt.legend()
        plt.show()
    return results_p0

beam_info_files_pattern = r"C:\SH Data Files\Beam Files\Target #3\10B_a_targ_test*.dat"

def histogram_energy():
    mpa_files_pattern_p0 = r"C:\SH Data Files\Data Files\Target #3\10B_a_targ_test_*.mpa"
    mpa_files_p0 = glob.glob(mpa_files_pattern_p0)
    slope = 0.001847
    intercept = 0.613362
    p0_energy_range = (3.5, 4.5)
    if not mpa_files_p0:
        print("No p0 files found!")
        results_p0 = []
    else:
        results_p0 = calculate_yield_plot_p0(mpa_files_p0, p0_energy_range, slope, intercept, beam_currents=[], yield_method="raw")
    return results_p0

if __name__ == "__main__":
    histogram_energy()
