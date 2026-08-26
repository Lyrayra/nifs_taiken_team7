import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

from gyaku import make_model

def get_raw_data(file_path, num_cols):
    data_list = []
    with open(file_path, 'r') as f:
        is_data = False
        for line in f:
            line = line.strip()
            if line.lower() == '[data]' or line.lower() == '#[data]' or line.lower() == '# [data]':
                is_data = True
                continue
            if is_data and line:
                parts = line.split(',')
                if len(parts) == num_cols:
                    data_list.append([float(v) for v in parts])
    return np.array(data_list)

def get_image_cube(raw_data):
    frames = int(raw_data[:, 0].max()) + 1
    x_max = int(raw_data[:, 1].max()) + 1
    y_max = int(raw_data[:, 2].max()) + 1
    image_cube = np.zeros((frames, y_max, x_max))
    for row in raw_data:
        frame_idx, x_idx, y_idx, count = int(row[0]), int(row[1]), int(row[2]), row[3]
        image_cube[frame_idx, y_idx, x_idx] = count
    return image_cube

print("Loading data...")
raw_neon = get_raw_data('a260825_img.txt', 4)
cube_neon = get_image_cube(raw_neon)
spec_neon = np.mean(cube_neon, axis=(0, 1))
x_max = cube_neon.shape[2]
x_indices = np.arange(x_max)
peak1_idx = 80 + np.argmax(spec_neon[80:105])

raw_plasma = get_raw_data('lhdcxs9a_img_sig@189129_t4.44s.txt', 13)
spectra_plasma = raw_plasma[:, 1:]

def evaluate_dispersion(disp):
    wavelength_axis = 528.03 + (x_indices - peak1_idx) * disp
    
    target_center_nm = 529.81891
    window_nm = 0.3
    
    # 装置関数 (ch1を使用)
    i_lambda_spec = np.mean(cube_neon[:, 0, :], axis=0)
    
    mask = (wavelength_axis >= target_center_nm - window_nm) & (wavelength_axis <= target_center_nm + window_nm)
    i_lambda = i_lambda_spec[mask]
    
    if len(i_lambda) < 5:
        return np.full(12, np.nan)
        
    edge_count = 5
    baseline = np.mean(np.concatenate([i_lambda[:edge_count], i_lambda[-edge_count:]]))
    i_lambda = i_lambda - baseline
    i_lambda = np.where(i_lambda < 0, 0, i_lambda)
    
    peak_idx = np.argmax(i_lambda)
    shift = len(i_lambda) // 2 - peak_idx
    i_lambda_centered = np.roll(i_lambda, shift)
    i_lambda_centered = i_lambda_centered / np.sum(i_lambda_centered)
    
    dvs = []
    for ch in range(12):
        y = spectra_plasma[:, ch]
        model = make_model(i_lambda_centered, target_center_nm, wavelength_axis)
        
        mask2 = ~np.isnan(y)
        x_fit = wavelength_axis[mask2]
        y_fit = y[mask2]
        
        A_guess = (np.max(y_fit) - np.min(y_fit)) * 1000
        p0 = [A_guess, 0.0, 50000.0, np.min(y_fit)]
        bounds = ([0, -1e6, 1000, -np.inf], [np.inf, 1e6, 1e7, np.inf])
        
        try:
            popt, _ = curve_fit(model, x_fit, y_fit, p0=p0, bounds=bounds)
            dvs.append(popt[2])
        except Exception:
            dvs.append(np.nan)
            
    return np.array(dvs)

if __name__ == '__main__':
    import pandas as pd
    # Data c の読み込み (data_n.csv)
    c_df = pd.read_csv('data_n.csv')
    c_R = c_df.iloc[:, 0].values
    c_dV = c_df.iloc[:, 1].values
    
    # a_R の読み込み (dv_r_profile.csv の 'a' の行)
    # 簡単のため、chに対応する a_R の近似値を手動で書くか、ファイルから読み込む
    import pandas as pd
    df = pd.read_csv('dv_r_profile.csv')
    a_R = df[df['Data'] == 'a']['R[m]'].values
    
    # c_dV を a_R の位置に補間する
    c_dV_interp = np.interp(a_R, c_R, c_dV)
    
    disps = np.linspace(-0.0250, -0.0350, 41)
    results = []
    
    print("Searching for dispersion that best matches Data c...")
    for disp in disps:
        dvs = evaluate_dispersion(disp)
        if np.any(np.isnan(dvs)):
            continue
            
        # Data c との誤差 (Mean Squared Error)
        error = np.mean((dvs - c_dV_interp)**2)
        results.append((disp, error, dvs))
        print(f"Dispersion {disp:.6f} -> MSE: {error:.2e}")
    
    results.sort(key=lambda x: x[1])
    best_disp, best_error, best_dvs = results[0]
    
    print(f"\nOptimal dispersion found: {best_disp:.6f}")
    print("Best dV profile:")
    print(np.round(best_dvs, 1))
    
    plt.figure(figsize=(10, 6))
    for disp, r, dvs in results[:5]:
        plt.plot(dvs, marker='o', label=f"disp={disp:.5f}")
    plt.legend()
    plt.title("Top 5 Smoothest dV Profiles")
    plt.xlabel("Channel")
    plt.ylabel("dV (m/s)")
    plt.grid(True)
    plt.savefig('opt_disp_result.png')
    print("Saved plot to opt_disp_result.png")
