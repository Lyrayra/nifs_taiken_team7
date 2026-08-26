import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
from scipy.interpolate import RegularGridInterpolator
import os

def load_mesh(mesh_file):
    """
    tsmeshファイルから (Z, R) とその点での実効マイナー半径 (reff) のデータを読み込む
    """
    z_list = []
    r_list = []
    reff_list = []
    
    with open(mesh_file, 'r') as f:
        is_data = False
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('#'):
                if '[Data]' in line:
                    is_data = True
                continue
            if is_data:
                parts = line.split(',')
                if len(parts) >= 7:
                    z = float(parts[0])
                    r = float(parts[1])
                    reff = float(parts[2])
                    z_list.append(z)
                    r_list.append(r)
                    reff_list.append(reff)
                    
    z_arr = np.array(z_list)
    r_arr = np.array(r_list)
    reff_arr = np.array(reff_list)
    
    # 2Dグリッドの作成
    z_grid = np.unique(z_arr)
    r_grid = np.unique(r_arr)
    
    reff_2d = np.zeros((len(z_grid), len(r_grid)))
    for z, r, reff in zip(z_arr, r_arr, reff_arr):
        iz = np.where(z_grid == z)[0][0]
        ir = np.where(r_grid == r)[0][0]
        reff_2d[iz, ir] = reff
        
    return z_grid, r_grid, reff_2d

def load_prep(prep_file):
    """
    prepファイルから チャンネル番号 (Signl) と メジャー半径 (R) などを読み込む
    """
    ch_list = []
    r_obs_list = []
    
    with open(prep_file, 'r') as f:
        is_data = False
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('#'):
                if '[Data]' in line:
                    is_data = True
                continue
            if is_data:
                parts = line.split(',')
                # ValNameに合わせたカラム位置 (ch=0, R=18)
                if len(parts) >= 19:
                    ch = int(parts[0])
                    r = float(parts[18])
                    ch_list.append(ch)
                    r_obs_list.append(r)
                    
    return np.array(ch_list), np.array(r_obs_list)

def plot_reff():
    # 対象ファイル
    mesh_file = 'tsmesh@189129_t4.44s_phi18deg.dat'
    prep_file = 'lhdcxs9a_prep@189129.dat'
    
    # 1. メッシュデータ(Z, R, reff)の読み込みと補間器の作成
    z_grid, r_grid, reff_2d = load_mesh(mesh_file)
    interpolator = RegularGridInterpolator((z_grid, r_grid), reff_2d, bounds_error=False, fill_value=np.nan)
    
    # 2. prepデータから観測位置(R)の読み込み
    ch_arr, r_obs_arr = load_prep(prep_file)
    
    # 3. 観測位置での reff を補間計算
    # 観測ポート（phi=18deg）は赤道面上(Z=0)と仮定して reff を取得します
    z_obs_arr = np.zeros_like(r_obs_arr) 
    
    pts = np.column_stack((z_obs_arr, r_obs_arr))
    reff_obs_arr = interpolator(pts)
    
    # プラズマ外（メッシュ内で10.0などダミー値が入っている部分）を除外
    reff_obs_arr[reff_obs_arr > 1.5] = np.nan
    
    # ターミナルへ値を出力
    print("Channel | R (m) | reff (ρ)")
    print("-" * 30)
    for ch, r, reff in zip(ch_arr, r_obs_arr, reff_obs_arr):
        print(f"   {ch:2d}   | {r:.3f} | {reff:.4f}")
        
    # 4. 図示
    fig, ax3 = plt.subplots(figsize=(8, 7))
    
    # プラズマ断面のヒートマップ（Z-R平面）と観測点
    # ダミー値を除外してプラズマ部分だけを抽出
    reff_plot = np.copy(reff_2d)
    reff_plot[reff_plot > 1.5] = np.nan
    
    R_mesh, Z_mesh = np.meshgrid(r_grid, z_grid)
    
    # reff（実効マイナー半径）は磁気軸より内側（低R側）などで負の値をとる場合があるため、
    # 等高線として閉じた面を描画するために絶対値をとります。
    abs_reff_plot = np.abs(reff_plot)
    
    cf = ax3.contourf(R_mesh, Z_mesh, abs_reff_plot, levels=np.linspace(0, 1.2, 25), cmap='plasma')
    cbar = plt.colorbar(cf, ax=ax3)
    cbar.set_label(r'Effective minor radius $|\rho|$')
    
    # プラズマ範囲の目安として |reff| = 0.6 の境界線を引く
    ax3.contour(R_mesh, Z_mesh, abs_reff_plot, levels=[0.6], colors='white', linewidths=2, linestyles='dashed')
    
    # 観測点のプロット
    ax3.scatter(r_obs_arr, z_obs_arr, color='cyan', edgecolors='black', s=40, label='Observation Points', zorder=5)
    for ch, r, z_obs in zip(ch_arr, r_obs_arr, z_obs_arr):
        # 点の横にチャンネル番号を付記
        ax3.text(r + 0.02, z_obs + 0.02, str(ch), color='white', fontsize=8, zorder=6,
                 path_effects=[patheffects.withStroke(linewidth=2, foreground='black')])
                 
    ax3.set_xlabel('Major Radius R (m)')
    ax3.set_ylabel('Vertical Position Z (m)')
    ax3.set_title('CXRS Channel Mapping & Plasma Cross Section')
    ax3.legend(loc='upper right')
    ax3.grid(True, linestyle=':', alpha=0.6)
    
    # R軸・Z軸のアスペクト比を揃える（物理的な形状を正しく表示するため）
    ax3.set_aspect('equal', 'box')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    plot_reff()
