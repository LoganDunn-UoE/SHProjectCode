"""

This code is used to establish the calibration parameters for future
analysis and visualisation.
It works as follows : 
    - file location is defined using the file_path variable
    - the approximate channel region in which peks are located is 
    specified using known_peaks with the energy of that peak also specifed.
    -Fits of the peak positions aswell as a calibration plot will be returned 

"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

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

def gaussian(x, a, x0, sigma):
    
    return a * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))

def calibrate_calcs(data, peaks):
 
    calibration_data = []
    region_width = 7  
    extra_margin = 50  
    for energy, approx_channel in peaks.items():
      
        fit_region = data[
            (data["Channel"] >= approx_channel - region_width) &
            (data["Channel"] <= approx_channel + region_width)]

        
        plot_region = data[
            (data["Channel"] >= approx_channel - region_width - extra_margin) &
            (data["Channel"] <= approx_channel + region_width + extra_margin)]

        try:
            popt, _ = curve_fit(
                gaussian,
                fit_region["Channel"],
                fit_region["Counts"],
                p0=[fit_region["Counts"].max(), approx_channel, 3]
            )

            fitted_channel = popt[1]  
            calibration_data.append((fitted_channel, energy))
            print(f"✔ Energy {energy} MeV fitted at channel {fitted_channel:.0f}")

           
            plt.figure(figsize=(8, 5))
            plt.bar(plot_region["Channel"], plot_region["Counts"], width=1.0, alpha=0.6, label="Histogram Data", color="gray", edgecolor="black")
            
           
            x_fit = np.linspace(plot_region["Channel"].min(), plot_region["Channel"].max(), 300)
            y_fit = gaussian(x_fit, *popt)
            plt.plot(x_fit, y_fit, label=f"Gaussian Fit (μ={popt[1]:.2f}, σ={popt[2]:.2f})", color="red", linewidth=2)

            plt.xlabel("Channel")
            plt.ylabel("Counts")
            plt.title(f"Gaussian Fit for {energy} MeV (±{extra_margin} Channels)")
            plt.legend()
            plt.grid(True)
            plt.show()

        except Exception as e:
            print(f" Peak fit error for {energy} MeV: {e}")

    if calibration_data:
        ch, en = zip(*calibration_data)
        coeffs = np.polyfit(ch, en, 1)
        slope, intercept = coeffs

        print(f"\n Calibration Equation: E(MeV) = {slope:.6f} * Channel + {intercept:.6f}")

    
        plt.figure(figsize=(8, 5))
        plt.scatter(ch, en, color='green', label='Calibration Data')
        
    
        x_min = 0
        x_max = max(ch) * 1.1  
        x_fit = np.linspace(x_min, x_max, 100)
        y_fit = np.polyval(coeffs, x_fit)

        plt.plot(x_fit, y_fit, label= f'\n Calibration Fit [E(MeV) = {slope:.6f}* Channel + {intercept:.6f}]', color="orange")

        plt.xlabel('Channel')
        plt.ylabel('Energy (MeV)')
        plt.title("Energy Calibration Fit")
        plt.legend()
        plt.grid(True)
        plt.show()

        return coeffs

def calibration():
    """
    Main function to load data, fit Gaussians to known peaks, and calibrate the detector.
    """
    file_path = r"C:\SH Data Files\10B_a_targ_test_099.mpa"
    data = load_mpa_file(file_path)

    if data is None:
        return

    known_peaks = {
        3.182: 1390,
          
        5.485: 2636 
    }

    calibrate_calcs(data, known_peaks)

calibration()
