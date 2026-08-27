"""
装置関数 (I(λ)) による「補正をしない」単純なガウス関数フィッティングと、
emcee で行った「補正済み」の解析結果を比較するスクリプト。
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import csv
import os

from sokudobunpu import generate_wavelength_distribution
from forward_model import load_dat_spectrum
from reff import load_prep

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def uncorrected_model(lam, A, v0, dV, bg, target_line):
    """
    装置関数の畳み込みを含まない純粋な理論スペクトルモデル。
    """
    return generate_wavelength_distribution(lam, A, target_line, v0, dV) + bg

def main():
    target_line = 529.81891
    
    # 観測データの読み込み
    dat_file = os.path.join(SCRIPT_DIR, 'lhdcxs9a_img_sig@189129_t4.44s.txt')
    dat_wl, dat_spectra = load_dat_spectrum(dat_file)
    num_channels = dat_spectra.shape[1]
    
    # prep ファイルを読み込んでチャンネル→主半径 R の対応を取得
    prep_file = os.path.join(SCRIPT_DIR, 'lhdcxs9a_prep@189129.dat')
    ch_arr, r_obs_arr = load_prep(prep_file)
    ch_to_r = {int(ch): r for ch, r in zip(ch_arr, r_obs_arr)}

    uncorrected_results = []
    
    print("【未補正フィッティング結果】")
    print(f"{'Ch':>3} | {'R (m)':>8} | {'dV (m/s)':>10} | {'dV_err':>10}")
    print("-" * 40)
    
    # チャンネルごとに未補正の単純フィッティングを実行
    for ch in range(num_channels):
        y = dat_spectra[:, ch]
        wl = dat_wl[:, ch]
        
        mask = ~np.isnan(y)
        x_fit = wl[mask]
        y_fit = y[mask]
        
        # フィット用ラッパー関数 (target_lineを固定)
        def fit_func(x, A, v0, dV, bg):
            return uncorrected_model(x, A, v0, dV, bg, target_line)
            
        A_guess = (np.max(y_fit) - np.min(y_fit)) * 1000
        bg_guess = np.min(y_fit)
        p0 = [A_guess, 0.0, 50000.0, bg_guess]
        bounds = ([0, -1e6, 1000, -np.inf], [np.inf, 1e6, 1e7, np.inf])
        
        try:
            popt, pcov = curve_fit(fit_func, x_fit, y_fit, p0=p0, bounds=bounds)
            A_fit, v0_fit, dV_fit, bg_fit = popt
            dV_err = np.sqrt(pcov[2, 2])
            
            R_val = ch_to_r.get(ch + 1, float(ch + 1))
            uncorrected_results.append({
                'ch': ch + 1,
                'R': R_val,
                'dV': dV_fit,
                'dV_err': dV_err,
                'error': False
            })
            print(f"{ch+1:>3} | {R_val:>8.4f} | {dV_fit:>10.1f} | {dV_err:>10.1f}")
        except Exception as e:
            print(f"Ch {ch+1} failed: {e}")
            uncorrected_results.append({'ch': ch + 1, 'error': True})
            
    # emceeの「補正済み」結果を読み込み
    emcee_R = []
    emcee_dV = []
    emcee_err_low = []
    emcee_err_high = []
    
    emcee_csv = os.path.join(SCRIPT_DIR, 'dv_r_profile_emcee.csv')
    try:
        with open(emcee_csv, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if not row: continue
                R_val = float(row[1])
                median = float(row[2])
                p16 = float(row[3])
                p84 = float(row[4])
                
                emcee_R.append(R_val)
                emcee_dV.append(median)
                emcee_err_low.append(median - p16)
                emcee_err_high.append(p84 - median)
    except FileNotFoundError:
        print(f"エラー: {emcee_csv} が見つかりません。先に gyaku_emcee.py を実行してください。")
        return

    # プロットして比較
    valid_uncorr = [r for r in uncorrected_results if not r['error']]
    R_uncorr = [r['R'] for r in valid_uncorr]
    dV_uncorr = [r['dV'] for r in valid_uncorr]
    dV_err_uncorr = [r['dV_err'] for r in valid_uncorr]

    plt.figure(figsize=(9, 6))
    
    # 未補正プロット
    plt.errorbar(R_uncorr, dV_uncorr, yerr=dV_err_uncorr, fmt='s-', color='red', 
                 markersize=8, capsize=5, capthick=1.5, elinewidth=1.5, 
                 label='Uncorrected (Simple Gaussian Fit)')
                 
    # 補正済み (emcee) プロット
    plt.errorbar(emcee_R, emcee_dV, yerr=[emcee_err_low, emcee_err_high], fmt='o-', color='blue', 
                 markersize=8, capsize=5, capthick=1.5, elinewidth=1.5, 
                 label='Corrected (MCMC with Instrumental Function)')

    plt.xlabel('Major Radius R (m)', fontsize=12)
    plt.ylabel('Thermal Width dV (m/s)', fontsize=12)
    plt.title('Effect of Instrumental Function Correction on Thermal Width', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=11)
    
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
