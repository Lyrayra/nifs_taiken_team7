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
    
    # 速度範囲の決定 (全チャンネルの v0 ± 4*dV をカバー)
    v_min = min(r['v0'] - 4 * r['dV'] for r in results)
    v_max = max(r['v0'] + 4 * r['dV'] for r in results)
    v = np.linspace(v_min, v_max, 500)
    
    # --- 図1: 各チャンネル個別のサブプロット ---
    ncols = 4
    nrows = (n_ch + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows), squeeze=False)
    axes_flat = axes.flatten()
    
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, n_ch))
    
    for i, res in enumerate(results):
        ax = axes_flat[i]
        fv = generate_velocity_distribution(v, res['A'], res['v0'], res['dV'])
        
        ax.plot(v / 1000.0, fv, color=colors[i], linewidth=2)
        ax.fill_between(v / 1000.0, fv, alpha=0.2, color=colors[i])
        ax.axvline(res['v0'] / 1000.0, color='red', linestyle=':', alpha=0.7, label=f"v0={res['v0']/1000:.1f} km/s")
        
        ax.set_title(f"Ch {res['ch']} (R={res['R']:.2f} m)", fontsize=11)
        ax.set_xlabel('Velocity (km/s)')
        ax.set_ylabel('f(v)')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, linestyle=':', alpha=0.5)
    
    # 余ったサブプロットを非表示
    for j in range(n_ch, len(axes_flat)):
        axes_flat[j].set_visible(False)
    
    fig.suptitle('Velocity Distribution f(v) for Each Channel', fontsize=14, y=1.02)
    fig.tight_layout()
    
    # --- 図2: 全チャンネル重ね合わせ ---
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    for i, res in enumerate(results):
        fv = generate_velocity_distribution(v, res['A'], res['v0'], res['dV'])
        # 形状比較のため面積を1に正規化
        fv_norm = fv / np.trapezoid(fv, v) if np.trapezoid(fv, v) > 0 else fv
        ax2.plot(v / 1000.0, fv_norm, color=colors[i], linewidth=1.5,
                 label=f"Ch{res['ch']} (R={res['R']:.2f}m, dV={res['dV']/1000:.0f}km/s)")
    
    ax2.set_xlabel('Velocity (km/s)', fontsize=12)
    ax2.set_ylabel('f(v) [normalized]', fontsize=12)
    ax2.set_title('Normalized Velocity Distributions (All Channels)', fontsize=14)
    ax2.legend(fontsize=9, loc='upper right')
    ax2.grid(True, linestyle='--', alpha=0.5)
    fig2.tight_layout()
    
    plt.show()


if __name__ == '__main__':
    plot_velocity_distributions()
