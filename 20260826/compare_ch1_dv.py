import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from ilambda_builder import make_i_lambda
from forward_model import load_dat_spectrum
from gyaku import make_model

def get_centered_i_lambda(file_path, target_peak_nm, channel=None):
    # 装置関数を生成し、中心にシフトして正規化する
    wl, i_lambda, _ = make_i_lambda(file_path, target_center_nm=target_peak_nm, window_nm=0.3, channel=channel)
    peak_idx = np.argmax(i_lambda)
    shift = len(i_lambda) // 2 - peak_idx
    i_lambda_centered = np.roll(i_lambda, shift)
    i_lambda_centered = i_lambda_centered / np.sum(i_lambda_centered)
    return i_lambda_centered

def fit_channel_1(i_lambda_centered, target_line, dat_wl, y):
    # 与えられた装置関数を使ってフィッティングを行う
    model = make_model(i_lambda_centered, target_line, dat_wl)
    
    mask = ~np.isnan(y)
    x_fit = dat_wl[mask]
    y_fit = y[mask]
    
    A_guess = (np.max(y_fit) - np.min(y_fit)) * 1000
    bg_guess = np.min(y_fit)
    v0_guess = 0.0
    dV_guess = 50000.0 
    
    p0 = [A_guess, v0_guess, dV_guess, bg_guess]
    bounds = ([0, -1e6, 1000, -np.inf], [np.inf, 1e6, 1e7, np.inf])
    
    popt, _ = curve_fit(model, x_fit, y_fit, p0=p0, bounds=bounds)
    return popt, model, x_fit, y_fit

def compare_ch1():
    target_line = 529.81891
    inst_file = 'a260825_img.txt'
    dat_file = 'lhdcxs9a_img_sig@189129_t4.44s.txt'
    
    # 観測データの読み込み (Ch1: インデックス 0)
    dat_wl, dat_spectra = load_dat_spectrum(dat_file)
    y_ch1 = dat_spectra[:, 0]
    
    # 1. 全チャンネル平均の装置関数を使用した場合のフィッティング
    i_lambda_avg = get_centered_i_lambda(inst_file, target_line, channel=None)
    popt_avg, model_avg, x_fit, y_fit = fit_channel_1(i_lambda_avg, target_line, dat_wl, y_ch1)
    
    # 2. Ch1固有の装置関数を使用した場合のフィッティング
    i_lambda_ch1 = get_centered_i_lambda(inst_file, target_line, channel=0)
    popt_ch1, model_ch1, _, _ = fit_channel_1(i_lambda_ch1, target_line, dat_wl, y_ch1)
    
    # 結果の出力
    print("【Ch1 (1ch) における dV の比較】")
    print(f"全チャンネル平均の装置関数を使用: dV = {popt_avg[2]:.2f} m/s")
    print(f"Ch1 固有の装置関数を使用        : dV = {popt_ch1[2]:.2f} m/s")
    print(f"差分                            : {popt_ch1[2] - popt_avg[2]:.2f} m/s")
    
    # プロット
    plt.figure(figsize=(10, 6))
    plt.plot(x_fit, y_fit, 'o', label='Observed Data (Ch1)', color='blue')
    
    # フィッティング曲線の生成 (x軸の表示用に元の間隔に合わせる)
    y_fit_avg = model_avg(x_fit, *popt_avg)
    y_fit_ch1 = model_ch1(x_fit, *popt_ch1)
    
    plt.plot(x_fit, y_fit_avg, '-', label=f'Fit (Avg Inst Func): dV={popt_avg[2]/1000:.1f} km/s', color='green', linewidth=2)
    plt.plot(x_fit, y_fit_ch1, '--', label=f'Fit (Ch1 Inst Func): dV={popt_ch1[2]/1000:.1f} km/s', color='red', linewidth=2)
    
    plt.xlabel('Wavelength (nm)', fontsize=12)
    plt.ylabel('Intensity', fontsize=12)
    plt.title('Comparison of Fitting Results for Ch1', fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    compare_ch1()
