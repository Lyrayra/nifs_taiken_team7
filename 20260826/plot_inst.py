import numpy as np
import matplotlib.pyplot as plt
from ilambda_builder import make_i_lambda

def plot_instrument_function():
    file_path = 'a260825_img.txt'
    target_peak_nm = 529.81891
    
    # 1. 装置関数 I(λ) の生成 (forward_model と同じパラメータ)
    wl, i_lambda, info = make_i_lambda(file_path, target_center_nm=target_peak_nm, window_nm=0.3)
    
    # 2. forward_model で行っているセンタリング処理 (コンボリューション用)
    peak_idx = np.argmax(i_lambda)
    shift = len(i_lambda) // 2 - peak_idx
    i_lambda_centered = np.roll(i_lambda, shift)
    
    # プロット
    plt.figure(figsize=(10, 5))
    
    # オリジナルの I(λ) と、センタリングされた I(λ) を比較
    plt.plot(wl, i_lambda, label='Original I(λ)', color='blue', linewidth=2, marker='o', markersize=4)
    plt.plot(wl, i_lambda_centered, label='Centered I(λ) (Used in Conv)', color='red', linestyle='--', linewidth=2, marker='x', markersize=5)
    
    # 最大値のピーク位置に線を引く
    plt.axvline(wl[peak_idx], color='blue', linestyle=':', alpha=0.5, label='Original Peak')
    plt.axvline(wl[len(i_lambda)//2], color='red', linestyle=':', alpha=0.5, label='Array Center')
    
    plt.xlabel('Wavelength (nm)', fontsize=12)
    plt.ylabel('Normalized Intensity', fontsize=12)
    plt.title(f'Instrument Function I(λ) extracted near {target_peak_nm} nm', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=11)
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    plot_instrument_function()
