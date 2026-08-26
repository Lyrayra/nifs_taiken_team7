import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
from scipy.optimize import curve_fit

# 他のスクリプトからのインポート
from sokudobunpu import generate_wavelength_distribution
from ilambda_builder import make_i_lambda
from forward_model import load_dat_spectrum

def make_model(i_lambda_centered, target_line, full_wl_axis):
    """
    scipy.optimize.curve_fit に渡すためのフィッティング関数（逆問題のモデル）を生成するファクトリ関数。
    
    引数:
    - i_lambda_centered: 中心化された装置関数 I(λ)
    - target_line: フィッティング対象の静止波長 (nm)
    - full_wl_axis: 等間隔な波長軸 (畳み込み演算を正しく行うため)
    """
    def model_func(x_subset, A, v0, dV, bg):
        # 1. 常にフル波長軸で理論スペクトル E(λ) を計算する
        e_lambda = generate_wavelength_distribution(full_wl_axis, A=A, lambda0=target_line, v0=v0, dV=dV)
        
        # 2. 畳み込み D(λ) = E(λ) * I(λ)
        d_lambda_full = convolve(e_lambda, i_lambda_centered, mode='same')
        
        # 3. バックグラウンドを加算
        d_lambda_total = d_lambda_full + bg
        
        # 4. curve_fit から渡された x_subset に対応する部分だけを補間して返す
        # 波長軸が降順(大きい順)の場合は np.interp が誤動作するため、昇順に反転させてから補間する
        if full_wl_axis[0] > full_wl_axis[-1]:
            return np.interp(x_subset, full_wl_axis[::-1], d_lambda_total[::-1])
        else:
            return np.interp(x_subset, full_wl_axis, d_lambda_total)
        
    return model_func

def solve_inverse_problem():
    # 対象のピーク波長
    inst_peak = 529.81891   # 装置関数 I(λ) を抽出するのに使うピーク
    target_line = 529.81891 # 今回フィッティングしたい輝線の静止波長
    
    # 2. 観測データ D(λ) の読み込み
    dat_file = 'lhdcxs9a_img_sig@189129_t4.44s.txt'
    dat_wl, dat_spectra = load_dat_spectrum(dat_file)
    num_channels = dat_spectra.shape[1]
    
    # フィット結果を保存するリスト
    results = []
    
    # 描画用のフィギュア設定
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    axes = axes.flatten()
    
    print("【逆問題 フィッティング結果】")
    print(f"{'Ch':>3} | {'v0 (m/s)':>10} | {'dV (m/s)':>10} | {'A':>10} | {'bg':>10}")
    print("-" * 55)
    
    for ch in range(num_channels):
        y = dat_spectra[:, ch]
        
        # チャンネルごとの波長軸
        wl = dat_wl[:, ch]
        
        # チャンネル固有の装置関数 I(λ) の生成
        _, i_lambda, _ = make_i_lambda('a260825_img.txt', target_center_nm=inst_peak, window_nm=0.3, channel=ch)
        
        # np.convolve の mode='same' のために、ピークを配列の中央に移動して正規化
        peak_idx = np.argmax(i_lambda)
        shift = len(i_lambda) // 2 - peak_idx
        i_lambda_centered = np.roll(i_lambda, shift)
        i_lambda_centered = i_lambda_centered / np.sum(i_lambda_centered)
        
        # 順問題モデル関数の作成 (このモデル関数には target_line と wl が固定で埋め込まれる)
        model = make_model(i_lambda_centered, target_line, wl)
        
        # 欠損値(NaN)を除去
        mask = ~np.isnan(y)
        x_fit = wl[mask]
        y_fit = y[mask]
        
        # 初期値 p0 = [A, v0, dV, bg] の推定
        # Aは最大カウント値などを参考に大雑把に設定、dVは熱速度幅の初期値(50km/s程度)
        A_guess = (np.max(y_fit) - np.min(y_fit)) * 1000
        bg_guess = np.min(y_fit)
        v0_guess = 0.0
        dV_guess = 50000.0 
        
        p0 = [A_guess, v0_guess, dV_guess, bg_guess]
        
        # 制約条件 bounds (最小値, 最大値)
        # 振幅 A > 0, v0 は制限なし, dV は最低1000m/s以上, bgは負の値も許容
        bounds = (
            [0, -1e6, 1000, -np.inf],
            [np.inf, 1e6, 1e7, np.inf]
        )
        
        try:
            # 曲線あてはめ実行
            popt, pcov = curve_fit(model, x_fit, y_fit, p0=p0, bounds=bounds)
            A_fit, v0_fit, dV_fit, bg_fit = popt
            
            results.append({
                'ch': ch + 1,
                'A': A_fit,
                'v0': v0_fit,
                'dV': dV_fit,
                'bg': bg_fit,
                'error': False
            })
            
            print(f"{ch+1:>3} | {v0_fit:>10.1f} | {dV_fit:>10.1f} | {A_fit:>10.1f} | {bg_fit:>10.1f}")
            
            # 各チャンネルのフィッティング結果を描画
            y_pred = model(x_fit, *popt)
            
            # forward_model と同様に、装置関数で広がる前の純粋な理論スペクトル E(λ) も描画
            e_lambda_raw = generate_wavelength_distribution(x_fit, A=A_fit, lambda0=target_line, v0=v0_fit, dV=dV_fit)
            e_lambda_plot = e_lambda_raw + bg_fit
            
            axes[ch].plot(x_fit, y_fit, 'o', color='blue', markersize=3, alpha=0.5, label='Data')
            axes[ch].plot(x_fit, e_lambda_plot, 'g--', label=r'Theory $E(\lambda)$')
            axes[ch].plot(x_fit, y_pred, 'r-', linewidth=2, label=r'Fit $D(\lambda)$')
            
            axes[ch].set_title(f'Ch {ch+1}: v0 = {v0_fit/1000:.1f} km/s')
            axes[ch].grid(True, linestyle=':', alpha=0.7)
            if ch == 0:
                axes[ch].legend(loc='best', fontsize='small')
                
        except Exception as e:
            print(f"Ch {ch+1} fitting failed: {e}")
            results.append({'ch': ch + 1, 'error': True})
            axes[ch].plot(x_fit, y_fit, 'o', color='gray', markersize=3, alpha=0.5)
            axes[ch].set_title(f'Ch {ch+1}: Fit Failed')
            axes[ch].grid(True, linestyle=':', alpha=0.7)
            
    plt.tight_layout()
    plt.suptitle('Inverse Problem: Fitting D(λ) = E(λ)*I(λ) + bg', fontsize=16, y=1.02)
    
    # 2枚目の図: チャンネルごとのプロファイル (v0 と dV)
    fig2, (ax_v0, ax_dv) = plt.subplots(1, 2, figsize=(14, 5))
    
    # prep ファイルを読み込んでチャンネル(ch)とメジャー半径(R)の対応を取得
    from reff import load_prep
    prep_file = 'lhdcxs9a_prep@189129.dat'
    ch_arr, r_obs_arr = load_prep(prep_file)
    ch_to_r = {int(ch): r for ch, r in zip(ch_arr, r_obs_arr)}
    
    ch_valid = [res['ch'] for res in results if not res['error']]
    r_valid = [ch_to_r.get(ch, ch) for ch in ch_valid] # 該当するRを取得
    v0_valid = [res['v0'] / 1000.0 for res in results if not res['error']] # km/s に変換
    dv_valid = [res['dV'] / 1000.0 for res in results if not res['error']] # km/s に変換
    
    # Bulk Velocity (v0)
    ax_v0.plot(r_valid, v0_valid, 'o-', color='green', linewidth=2, markersize=8)
    ax_v0.axhline(0, color='black', linestyle='-', alpha=0.3)
    ax_v0.set_xlabel('Major Radius R (m)')
    ax_v0.set_ylabel('Bulk Velocity v0 (km/s)')
    ax_v0.set_title('Extracted Bulk Velocity Profile')
    ax_v0.grid(True, linestyle='--')
    
    # Thermal Width (dV)
    ax_dv.plot(r_valid, dv_valid, 's-', color='orange', linewidth=2, markersize=8)
    ax_dv.set_xlabel('Major Radius R (m)')
    ax_dv.set_ylabel('Thermal Width dV (km/s)')
    ax_dv.set_title('Extracted Thermal Width Profile')
    ax_dv.grid(True, linestyle='--')
    
    fig2.tight_layout()
    
    # CSVファイルへの保存
    import csv
    csv_filename = 'dv_r_profile.csv'
    with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # dv を m/s で出力するようにヘッダーを変更
        writer.writerow(['Channel', 'R (m)', 'v0 (km/s)', 'dV (m/s)'])
        for ch, r, v0, dv in zip(ch_valid, r_valid, v0_valid, dv_valid):
            # dv_valid は km/s なので 1000 を掛けて m/s に戻す
            writer.writerow([ch, f"{r:.4f}", f"{v0:.4f}", f"{dv * 1000.0:.4f}"])
    print(f"\n[INFO] プロファイルデータを '{csv_filename}' として保存しました。")
    
    plt.show()

if __name__ == '__main__':
    solve_inverse_problem()
