import numpy as np
import matplotlib.pyplot as plt
from ilambda_builder import make_i_lambda

def plot_instrument_function_ch1():
    file_path = 'a260825_img.txt'
    target_peak_nm = 529.81891
    
    # チャンネル1 (0インデックスなので0) を指定して装置関数を生成
    target_channel = 0
    
    wl, i_lambda, info = make_i_lambda(file_path, target_center_nm=target_peak_nm, window_nm=0.3, channel=target_channel)
    
    # コンボリューション用のセンタリング処理
    peak_idx = np.argmax(i_lambda)
    shift = len(i_lambda) // 2 - peak_idx
    i_lambda_centered = np.roll(i_lambda, shift)
    
    # プロット
    plt.figure(figsize=(10, 5))
    
    plt.plot(wl, i_lambda, label='Original I(λ) for 1ch', color='blue', linewidth=2, marker='o', markersize=4)
    plt.plot(wl, i_lambda_centered, label='Centered I(λ) for 1ch', color='red', linestyle='--', linewidth=2, marker='x', markersize=5)
    
    plt.axvline(wl[peak_idx], color='blue', linestyle=':', alpha=0.5, label='Original Peak')
    plt.axvline(wl[len(i_lambda)//2], color='red', linestyle=':', alpha=0.5, label='Array Center')
    
    plt.xlabel('Wavelength (nm)', fontsize=12)
    plt.ylabel('Normalized Intensity', fontsize=12)
    plt.title(f'Instrument Function I(λ) for Channel 1 near {target_peak_nm} nm', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=11)
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    plot_instrument_function_ch1()
