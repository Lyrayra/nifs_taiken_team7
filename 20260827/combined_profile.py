import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import matplotlib.patheffects as patheffects
import csv
import os
from scipy.interpolate import RegularGridInterpolator
from reff import load_mesh, load_prep

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def load_temperature_data():
    # --- aのデータ ---
    a_R, a_dV, a_dV_err = [], [], []
    csv_file = os.path.join(SCRIPT_DIR, 'dv_r_profile.csv')
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if not row: continue
                a_R.append(float(row[1]))
                a_dV.append(float(row[3]))
                a_dV_err.append(float(row[4]) if len(row) > 4 else 0.0)
    except FileNotFoundError:
        pass
        
    # --- bのデータ (画像から書き起こしたもの + MCMCの16%と84%) ---
    # format: [ch, R, dV_curvefit, dV_err_curvefit, dV_mcmc_median, dV_mcmc_val1, dV_mcmc_val2]
    b_data_full = np.array([
        [0, 3.62962, 192439.9877939988, 11925.03505374918, 192831.22531609, 185320.10567832, 201220.60878072],
        [1, 3.6939, 245616.06530712068, 10457.351299181802, 245885.73367483, 236220.89358393, 256555.21675301],
        [2, 3.75868, 248914.7592786996, 10025.693074786484, 249483.19271859, 240590.27187963, 258785.81061652],
        [3, 3.83491, 233249.58298758548, 5966.759459465759, 233793.03655107, 228990.6896344, 238375.90888207],
        [4, 3.93396, 203064.91313932854, 4063.2225913846432, 203206.8852716, 200460.52172928, 205906.95218869],
        [5, 4.0342, 203676.21002683835, 2804.9898767972054, 203818.25213482, 202038.60238968, 205566.84252004],
        [6, 4.13564, 172496.9011050146, 2249.8181173319213, 172311.86194598, 178316.84523549, 166893.7182433],
        [7, 4.23832, 164262.26199249073, 1364.0632902031, 164243.81361714, 168892.72029279, 159045.35706482],
        [8, 4.34225, 136953.750982935, 1094.4103715900956, 136366.02291598, 140285.25241522, 132632.82180607],
        [9, 4.4357, 126657.08477723994, 930.4047534014084, 127857.26770743, 130716.53073338, 124576.37223675],
        [10, 4.50646, 108573.06372809905, 870.8415709304451, 108356.22915676, 110589.44309276, 106033.37396945],
        [11, 4.5778, 95828.82416929209, 1644.6870257624037, 95949.04198586, 94051.25885956, 97624.59611614]
    ])
    b_R = b_data_full[:, 1]
    b_dV = b_data_full[:, 2]
    b_dV_err = b_data_full[:, 3]
    b_dV_16 = np.minimum(b_data_full[:, 5], b_data_full[:, 6])
    b_dV_84 = np.maximum(b_data_full[:, 5], b_data_full[:, 6])
    
    # --- cのデータ ---
    c_R, c_dV, c_dV_err = [], [], []
    try:
        data_n_file = os.path.join(SCRIPT_DIR, 'data_n.csv')
        with open(data_n_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if not row: continue
                c_R.append(float(row[0]))
                c_dV.append(float(row[1]))
                c_dV_err.append(float(row[2]) if len(row) > 2 else 0.0)
    except FileNotFoundError:
        pass
        
    # --- MCMC帯データ (emcee) ---
    emcee_R, emcee_dV_16, emcee_dV_84 = [], [], []
    try:
        emcee_csv = os.path.join(SCRIPT_DIR, 'dv_r_profile_emcee.csv')
        with open(emcee_csv, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if row:
                    emcee_R.append(float(row[1]))
                    emcee_dV_16.append(float(row[3]))
                    emcee_dV_84.append(float(row[4]))
    except: pass
    
    # mc_R, mc_dV_16, mc_dV_84 はダミー（削除済みのため）
    mc_R, mc_dV_16, mc_dV_84 = [], [], []
    
    # 単位変換 -> T (K)
    AMU = 1.660539e-27
    M_C = 12.0 * AMU
    K_B = 1.380649e-23
    def dV_to_T(dv): return 0.5 * M_C * (dv ** 2) / K_B
    def dV_err_to_T_err(dv, dv_err): return M_C * dv * dv_err / K_B
    
    a_dV_err = dV_err_to_T_err(np.array(a_dV), np.array(a_dV_err))
    a_dV = dV_to_T(np.array(a_dV))
    a_R = np.array(a_R)
    
    b_dV_err = dV_err_to_T_err(np.array(b_dV), np.array(b_dV_err))
    b_dV = dV_to_T(np.array(b_dV))
    b_dV_16 = dV_to_T(np.array(b_dV_16))
    b_dV_84 = dV_to_T(np.array(b_dV_84))
    
    c_dV_err = dV_err_to_T_err(np.array(c_dV), np.array(c_dV_err))
    c_dV = dV_to_T(np.array(c_dV))
    c_R = np.array(c_R)
    
    emcee_R = np.array(emcee_R)
    emcee_dV_16 = dV_to_T(np.array(emcee_dV_16))
    emcee_dV_84 = dV_to_T(np.array(emcee_dV_84))
    
    mc_R = np.array(mc_R)
    mc_dV_16 = dV_to_T(np.array(mc_dV_16))
    mc_dV_84 = dV_to_T(np.array(mc_dV_84))
    
    # 未補正データ
    uncorr_R, uncorr_dV, uncorr_dV_err = [], [], []
    try:
        with open(os.path.join(SCRIPT_DIR, 'uncorrected_dv.csv'), 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if row:
                    uncorr_R.append(float(row[1]))
                    uncorr_dV.append(float(row[2]))
                    uncorr_dV_err.append(float(row[3]))
        uncorr_R = np.array(uncorr_R)
        uncorr_dV = np.array(uncorr_dV)
        uncorr_dV_err = np.array(uncorr_dV_err)
        # threedv_T.py と同じ順序: 先に誤差変換 (生の dV を使う)、後に dV を温度変換
        uncorr_dV_err = dV_err_to_T_err(uncorr_dV, uncorr_dV_err)
        uncorr_dV = dV_to_T(uncorr_dV)
    except: pass
    
    return {
        'a': (a_R, a_dV, a_dV_err),
        'b': (b_R, b_dV, b_dV_err, b_dV_16, b_dV_84),
        'c': (c_R, c_dV, c_dV_err),
        'emcee': (emcee_R, emcee_dV_16, emcee_dV_84),
        'mc': (mc_R, mc_dV_16, mc_dV_84),
        'uncorr': (uncorr_R, uncorr_dV, uncorr_dV_err)
    }

def main():
    # 1. データ読み込み
    data = load_temperature_data()
    mesh_file = os.path.join(SCRIPT_DIR, 'tsmesh@189129_t4.44s_phi18deg.dat')
    z_grid, r_grid, reff_2d = load_mesh(mesh_file)
    
    a_R, a_dV, a_dV_err = data['a']
    b_R, b_dV, b_dV_err, b_dV_16, b_dV_84 = data['b']
    c_R, c_dV, c_dV_err = data['c']
    emcee_R, emcee_dV_16, emcee_dV_84 = data['emcee']
    mc_R, mc_dV_16, mc_dV_84 = data['mc']
    uncorr_R, uncorr_dV, uncorr_dV_err = data['uncorr']
    
    # 2. サブプロットの作成 (上: 温度, 下: 断面)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True, gridspec_kw={'height_ratios': [1.2, 1]})
    plt.subplots_adjust(hspace=0.05)
    
    # ----------------------------------------------------
    # 上部: 温度プロファイル (threedv_T.pyと同じ見え方にする)
    # ----------------------------------------------------
    all_R = np.concatenate([a_R, b_R, c_R])
    all_dV = np.concatenate([a_dV, b_dV, c_dV])
    sort_idx = np.argsort(all_R)
    
    # Connected Path
    ax1.plot(all_R[sort_idx], all_dV[sort_idx], '-', color='black', linewidth=1.2, alpha=0.5, zorder=1, label='Connected Path')
    
    # データプロット (エラーバー付き)
    ax1.errorbar(a_R, a_dV, yerr=a_dV_err, fmt='o', color='orange', markersize=7, 
                 capsize=4, elinewidth=1.2, capthick=1.2, label='Data a', zorder=2)
    ax1.errorbar(b_R, b_dV, yerr=b_dV_err, fmt='^', color='green', markersize=7, 
                 capsize=4, elinewidth=1.2, capthick=1.2, label='Data b', zorder=2)
    ax1.errorbar(c_R, c_dV, yerr=c_dV_err, fmt='s', color='blue', markersize=7, 
                 capsize=4, elinewidth=1.2, capthick=1.2, label='Data c', zorder=2)
                 
    # 未補正データ
    if len(uncorr_R) > 0:
        ax1.errorbar(uncorr_R, uncorr_dV, yerr=uncorr_dV_err, fmt='D-', color='gray', markersize=6, 
                     capsize=4, elinewidth=1.2, capthick=1.2, label='Uncorrected (Simple Fit)', zorder=1)
    
    # MCMC 帯
    all_arrays_R = [emcee_R, b_R, mc_R]
    all_arrays_16 = [emcee_dV_16, b_dV_16, mc_dV_16]
    all_arrays_84 = [emcee_dV_84, b_dV_84, mc_dV_84]
    
    valid_arrays_R = [arr for arr in all_arrays_R if len(arr) > 0]
    valid_arrays_16 = [arr for arr in all_arrays_16 if len(arr) > 0]
    valid_arrays_84 = [arr for arr in all_arrays_84 if len(arr) > 0]
    
    combined_mc_R = np.concatenate(valid_arrays_R) if len(valid_arrays_R) > 0 else np.array([])
    combined_mc_dV_16 = np.concatenate(valid_arrays_16) if len(valid_arrays_16) > 0 else np.array([])
    combined_mc_dV_84 = np.concatenate(valid_arrays_84) if len(valid_arrays_84) > 0 else np.array([])
    
    if len(combined_mc_R) > 0:
        sort_mc_idx = np.argsort(combined_mc_R)
        sorted_mc_R = combined_mc_R[sort_mc_idx]
        sorted_mc_dV_16 = combined_mc_dV_16[sort_mc_idx]
        sorted_mc_dV_84 = combined_mc_dV_84[sort_mc_idx]
        
        ax1.fill_between(sorted_mc_R, sorted_mc_dV_16, sorted_mc_dV_84, 
                         color='purple', alpha=0.15, zorder=2, label='MCMC 16%-84% Range')
    
    ax1.set_ylabel('Temperature $T = m dV^2 / (2 k_B)$ (K)', fontsize=12)
    ax1.set_title('Combined View: Temperature Profile & Plasma Cross Section', fontsize=14)
    ax1.grid(True, linestyle=':', alpha=0.7)
    
    # threedv_T.pyに合わせてスケールと凡例を設定
    ax1.set_ylim(0, 5.2e7)
    ax1.legend(loc='upper right', fontsize=11)
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    # ----------------------------------------------------
    # 下部: 断面マップ (右半分だけ: R >= 3.0 あたりから)
    # ----------------------------------------------------
    reff_plot = np.copy(reff_2d)
    reff_plot[reff_plot > 1.5] = np.nan
    abs_reff_plot = np.abs(reff_plot)
    R_mesh, Z_mesh = np.meshgrid(r_grid, z_grid)
    
    cf = ax2.contourf(R_mesh, Z_mesh, abs_reff_plot, levels=np.linspace(0, 1.2, 25), cmap='plasma')
    cbar = plt.colorbar(cf, ax=ax2, orientation='horizontal', pad=0.15, aspect=40)
    cbar.set_label(r'Effective minor radius $|\rho|$')
    ax2.contour(R_mesh, Z_mesh, abs_reff_plot, levels=[0.6], colors='white', linewidths=2, linestyles='dashed')
    
    # 各データの観測点を Z=0 にプロット
    ax2.scatter(a_R, np.zeros_like(a_R), color='orange', marker='o', edgecolors='black', s=50, zorder=5)
    ax2.scatter(b_R, np.zeros_like(b_R), color='green', marker='^', edgecolors='black', s=50, zorder=5)
    ax2.scatter(c_R, np.zeros_like(c_R), color='blue', marker='s', edgecolors='black', s=50, zorder=5)
    
    # チャンネル番号は Data a のみ (見づらくなるため)
    prep_file = os.path.join(SCRIPT_DIR, 'lhdcxs9a_prep@189129.dat')
    ch_arr, _ = load_prep(prep_file)
    for ch, r in zip(ch_arr, a_R):
        ax2.text(r + 0.01, 0 + 0.01, str(ch), color='white', fontsize=8, zorder=6,
                 path_effects=[patheffects.withStroke(linewidth=2, foreground='black')])
                 
    ax2.set_xlabel('Major Radius R (m)', fontsize=12)
    ax2.set_ylabel('Z (m)', fontsize=12)
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    # ----------------------------------------------------
    # 表示範囲と縦線（点と点を結ぶ）の設定
    # ----------------------------------------------------
    # 表示範囲の制限 (右半分)
    # 温度データの R は 3.6 ~ 4.6 なので、少し広めにとる
    min_R = min(combined_mc_R) - 0.2 if len(combined_mc_R) > 0 else 3.4
    max_R = max(combined_mc_R) + 0.2 if len(combined_mc_R) > 0 else 4.8
    ax2.set_xlim(min_R, max_R)
    
    # 表示範囲の制限 (右半分)
    
    # Z方向を適度に絞る
    ax2.set_ylim(-0.5, 0.5) 
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
