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

def load_dat_spectrum(file_path):
    import numpy as np
    data_list = []
    with open(file_path, 'r') as f:
        is_data_section = False
        for line in f:
            line = line.strip()
            if line == '[Data]' or line == '# [Data]':
                is_data_section = True
                continue
            
            if is_data_section and line and not line.startswith('#'):
                parts = [float(v) for v in line.split(',')]
                data_list.append(parts)
                
    raw_data = np.array(data_list)
    
    # 0 column is xpix, 1-12 columns are stripes
    spectra_data = raw_data[:, 1:] # Return all 12 stripes without averaging
    
    # Wavelength axis
    x_max = len(spectra_data)
    pixel_peaks = np.array([12, 33, 92, 111])
    wavelength_peaks = np.array([530.47580, 529.81891, 528.03, 527.40393])
    poly = np.poly1d(np.polyfit(pixel_peaks, wavelength_peaks, 1))
    wavelength_axis = np.linspace(poly(0), poly(x_max - 1), x_max)
    
    return wavelength_axis, spectra_data

def plot_forward_model():
    import os
    file_path = os.path.join(os.path.dirname(__file__), 'a260825_img.txt')
    
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
    
    import os
    # 2つ目のデータ (.txt) を読み込む
    dat_file_path = os.path.join(os.path.dirname(__file__), 'lhdcxs9a_img_sig@189129_t4.44s.txt')
    dat_wl, dat_spectra = load_dat_spectrum(dat_file_path)

    # 新しいデータ (.txt) のスケールを畳み込みの山の頂点に合わせる
    # チャンネル間の相対的な強さを保つため、全チャンネルの最大・最小値を使って一括でスケーリングします
    d_max = np.nanmax(d_spec)
    global_dat_max = np.nanmax(dat_spectra)
    global_dat_min = np.nanmin(dat_spectra)
    
    # プロット
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    ax1.plot(wl, m_spec, label=r'Measured $M(\lambda)$', color='blue', alpha=0.5)
    ax1.plot(wl, e_spec, label=r'Theory $E(\lambda)$ (No Inst. Broadening)', color='green', linestyle='--')
    ax1.plot(wl, d_spec, label=r'Forward Model $D(\lambda) = E*I + bg$', color='red', linewidth=2)
    
    # 12個のチャンネルをそれぞれ描画
    num_channels = dat_spectra.shape[1]
    import matplotlib.cm as cm
    colors = cm.turbo(np.linspace(0, 1, num_channels)) # チャンネルごとに色を分ける
    
    for ch in range(num_channels):
        dat_spec = dat_spectra[:, ch]
        if global_dat_max > global_dat_min:
            dat_spec_scaled = (dat_spec - global_dat_min) / (global_dat_max - global_dat_min) * (d_max - bg) + bg
        else:
            dat_spec_scaled = dat_spec + bg
            
        ax1.plot(dat_wl, dat_spec_scaled, color=colors[ch], linewidth=1.2, alpha=0.8, label=f'Ch {ch+1}')
    
    # 解析対象のピークだけでなく、新しいデータの山も見えるように全範囲を表示
    # ax1.set_xlim(target_line - 0.5, target_line + 0.5)
    # ax1.set_ylim(bg - 50, np.nanmax(m_spec) * 1.1)

    ax1.set_xlabel('Wavelength (nm)')
    ax1.set_ylabel('Count / Scaled Signal')
    plt.title('Forward Problem: Model vs Measurement')
    
    # 凡例を外側に配置して見やすくする
    ax1.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize='small', borderaxespad=0.)
    ax1.grid(True, linestyle=':')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_forward_model()
