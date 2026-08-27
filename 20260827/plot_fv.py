"""
各チャンネルの速度分布関数 f(v) をプロットするスクリプト。
gyaku.py のフィッティング結果 (dv_r_profile.csv) を読み込み、
ガウス型速度分布を速度空間で描画する。
"""
import numpy as np
import matplotlib.pyplot as plt
import csv
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 物理定数
C_MS = 299792458.0  # 光速 (m/s)


def generate_velocity_distribution(v, A, v0, dV):
    """ガウス型速度分布関数 f(v)"""
    return (A / (np.sqrt(np.pi) * dV)) * np.exp(-((v - v0) / dV)**2)


def load_fit_results(csv_file):
    """dv_r_profile.csv からフィッティング結果を読み込む"""
    results = []
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row['Channel']:
                continue
            results.append({
                'ch': int(row['Channel']),
                'R': float(row['R (m)']),
                'v0': float(row['v0 (km/s)']) * 1000.0,  # km/s -> m/s
                'dV': float(row['dV (m/s)']),
                'dV_err': float(row.get('dV_err (m/s)', 0)),
                'A': float(row['A']),
            })
    return results


def plot_velocity_distributions():
    csv_file = os.path.join(SCRIPT_DIR, 'dv_r_profile.csv')
    
    try:
        results = load_fit_results(csv_file)
    except FileNotFoundError:
        print(f"エラー: {csv_file} が見つかりません。先に gyaku.py を実行してください。")
        return
    except KeyError as e:
        print(f"エラー: CSVに列 {e} がありません。gyaku.py を再実行してCSVを更新してください。")
        return
    
    n_ch = len(results)
    if n_ch == 0:
        print("フィッティング結果がありません。")
        return
    
    # --- R vs dV (エラーバー付き) ---
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    
    r_arr = [res['R'] for res in results]
    dv_arr = [res['dV'] / 1000.0 for res in results]          # km/s
    dv_err_arr = [res['dV_err'] / 1000.0 for res in results]  # km/s
    
    ax3.errorbar(r_arr, dv_arr, yerr=dv_err_arr, fmt='s-',
                 color='orange', markersize=8, linewidth=2,
                 capsize=5, capthick=1.5, ecolor='gray', elinewidth=1.5,
                 label='dV ± 1σ')
    ax3.set_xlabel('Major Radius R (m)', fontsize=12)
    ax3.set_ylabel('Thermal Width dV (km/s)', fontsize=12)
    ax3.set_title('R - dV Profile with Error Bars', fontsize=14)
    ax3.legend(fontsize=11)
    ax3.grid(True, linestyle='--', alpha=0.5)
    fig3.tight_layout()
    
    plt.show()


if __name__ == '__main__':
    plot_velocity_distributions()
