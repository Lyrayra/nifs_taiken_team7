import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
from sokudobunpu import generate_wavelength_distribution, load_average_spectrum
from ilambda_builder import make_i_lambda

def solve_forward_problem(file_path, target_peak_nm, lambda0, v0_param, dV_param, A_param, background):
    """
    順問題を解いて、モデルスペクトル D(λ) を計算する。
    
    引数:
    - file_path: 実測データのパス
    - target_peak_nm: 装置関数 I(λ) を抽出するための基準ピーク波長
    - lambda0: 解析対象とする輝線の静止波長
    - v0_param: バルク速度 (m/s)
    - dV_param: 熱速度幅 (m/s)
    - A_param: 輝線の強度パラメータ
    - background: バックグラウンドのカウント値
    """
    # 1. 観測データの読み込み (比較対象として使うため)
    measured_wavelength_axis, measured_spectrum, _ = load_average_spectrum(file_path)
    
    # 2. 装置関数 I(λ) の生成 (ターゲットピークの周辺一山分だけから切り出し)
    # ilambda_builder を使って、指定したピーク周辺 ±0.3nm から I(λ) を抽出
    _, i_lambda, _ = make_i_lambda(file_path, target_center_nm=target_peak_nm, window_nm=0.3)
    
    # np.convolve の mode='same' を使ったときに位置がずれないよう、
    # i_lambda のピークが配列のド真ん中に来るようにシフト（センタリング）する
    peak_idx = np.argmax(i_lambda)
    shift = len(i_lambda) // 2 - peak_idx
    i_lambda_centered = np.roll(i_lambda, shift)
    
    # 畳み込み時にエネルギーが保存されるよう、和が1になるように正規化
    i_lambda_centered = i_lambda_centered / np.sum(i_lambda_centered)
    
    # 3. 理論スペクトル E(λ) の生成
    # 実測データの波長軸全体に対して E(λ) を計算する
    e_lambda = generate_wavelength_distribution(
        measured_wavelength_axis, 
        A=A_param, 
        lambda0=lambda0, 
        v0=v0_param, 
        dV=dV_param
    )
    
    # 4. 畳み込み D(λ) = E(λ) * I(λ)
    d_lambda = convolve(e_lambda, i_lambda_centered, mode='same')
    
    # バックグラウンドを加算して最終的な順問題の出力とする
    d_lambda_total = d_lambda + background
    
    return measured_wavelength_axis, measured_spectrum, d_lambda_total, e_lambda + background

def plot_forward_model():
    file_path = 'a260825_img.txt'
    
    # 【パラメータの設定】
    inst_peak = 529.81891   # 装置関数 I(λ) を切り出すのに使う綺麗なピーク
    target_line = 529.81891 # 今回モデル化（フィッティング）したい輝線の静止波長
    
    # 仮の物理パラメータ（のちに逆問題で最適化する対象）
    v0 = 0.0          # バルク速度 (m/s)
    dV = 100000.0     # 熱速度の広がり (m/s) - カクカクしないよう大きめに設定
    A = 3000.0        # 振幅（カウントのスケール）- ピーク高さが12000付近になるよう調整
    bg = 2100.0       # バックグラウンド - グラフのベースラインに合わせる
    
    # 順問題を解く
    wl, m_spec, d_spec, e_spec = solve_forward_problem(
        file_path, 
        target_peak_nm=inst_peak, 
        lambda0=target_line, 
        v0_param=v0, 
        dV_param=dV, 
        A_param=A, 
        background=bg
    )
    
    # プロット
    plt.figure(figsize=(10, 5))
    plt.plot(wl, m_spec, label='Measured $M(\lambda)$', color='blue', alpha=0.5)
    plt.plot(wl, e_spec, label='Theory $E(\lambda)$ (No Inst. Broadening)', color='green', linestyle='--')
    plt.plot(wl, d_spec, label='Forward Model $D(\lambda) = E*I + bg$', color='red', linewidth=2)
    
    # 解析対象のピーク周辺を拡大表示
    plt.xlim(target_line - 0.5, target_line + 0.5)
    plt.ylim(bg - 50, np.nanmax(m_spec[(wl > target_line - 0.5) & (wl < target_line + 0.5)]) * 1.1)
    
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Count')
    plt.title('Forward Problem: Model vs Measurement')
    plt.legend()
    plt.grid(True, linestyle=':')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_forward_model()
