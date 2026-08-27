import numpy as np
import matplotlib.pyplot as plt
import csv
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def plot_comparison():
    # --- aのデータ (今回 gyaku.py / gosa.py で計算して保存したもの) を読み込む ---
    a_R = []
    a_dV = []
    a_dV_err = []
    
    csv_file = os.path.join(SCRIPT_DIR, 'dv_r_profile.csv')
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader) # ヘッダーをスキップ
            for row in reader:
                if not row: continue
                # 1列目がR, 3列目がdV, 4列目がdV_err (存在する場合)
                a_R.append(float(row[1]))
                a_dV.append(float(row[3])) # m/s のまま
                if len(row) > 4:
                    a_dV_err.append(float(row[4]))
                else:
                    a_dV_err.append(0.0)
    except FileNotFoundError:
        print(f"エラー: {csv_file} が見つかりません。先に gyaku.py または gosa.py を実行してください。")
        return
            
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
    
    # --- cのデータ (data_n.csv) ---
    c_R = []
    c_dV = []
    c_dV_err = []
    try:
        data_n_file = os.path.join(SCRIPT_DIR, 'data_n.csv')
        with open(data_n_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader) # ヘッダーをスキップ
            for row in reader:
                if not row: continue
                c_R.append(float(row[0]))
                c_dV.append(float(row[1])) # m/s のまま
                if len(row) > 2:
                    c_dV_err.append(float(row[2]))
                else:
                    c_dV_err.append(0.0)
        c_R = np.array(c_R)
        c_dV = np.array(c_dV)
        c_dV_err = np.array(c_dV_err)
    except FileNotFoundError:
        print(f"エラー: data_n.csv が見つかりません。")
        return
    # --- MCMC のデータ a (dv_r_profile_emcee.csv) を線の追加用として読み込み ---
    emcee_R = []
    emcee_dV_16 = []
    emcee_dV_84 = []
    try:
        emcee_file = os.path.join(SCRIPT_DIR, 'dv_r_profile_emcee.csv')
        with open(emcee_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if not row: continue
                emcee_R.append(float(row[1]))
                emcee_dV_16.append(float(row[3]))
                emcee_dV_84.append(float(row[4]))
        emcee_R = np.array(emcee_R)
        emcee_dV_16 = np.array(emcee_dV_16)
        emcee_dV_84 = np.array(emcee_dV_84)
    except FileNotFoundError:
        print(f"エラー: dv_r_profile_emcee.csv が見つかりません。")

    # --- MCMC のデータ c (montecarlo_results.csv) を線の追加用として読み込み ---
    mc_R = []
    mc_dV_16 = []
    mc_dV_84 = []
    try:
        mc_file = os.path.join(SCRIPT_DIR, 'montecarlo_results.csv')
        with open(mc_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader) # ヘッダーをスキップ
            for row in reader:
                if not row: continue
                mc_R.append(float(row[0]))
                mc_dV_16.append(float(row[2])) # dv_lower
                mc_dV_84.append(float(row[3])) # dv_upper
        mc_R = np.array(mc_R)
        mc_dV_16 = np.array(mc_dV_16)
        mc_dV_84 = np.array(mc_dV_84)
    except FileNotFoundError:
        print(f"エラー: montecarlo_results.csv が見つかりません。")
        
    # --- 未補正データ (uncorrected_dv.csv) ---
    uncorr_R = []
    uncorr_dV = []
    uncorr_dV_err = []
    try:
        uncorr_file = os.path.join(SCRIPT_DIR, 'uncorrected_dv.csv')
        with open(uncorr_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if not row: continue
                uncorr_R.append(float(row[1]))
                uncorr_dV.append(float(row[2]))
                uncorr_dV_err.append(float(row[3]))
        uncorr_R = np.array(uncorr_R)
        uncorr_dV = np.array(uncorr_dV)
        uncorr_dV_err = np.array(uncorr_dV_err)
    except FileNotFoundError:
        print(f"エラー: uncorrected_dv.csv が見つかりません。")
    
    # ========================================================
    # 単位変換: dV (m/s) -> T (K)
    # T = m * dV^2 / k_B
    # 誤差伝播: dT = 2 * m * dV * dV_err / k_B
    # ========================================================
    AMU = 1.660539e-27  # kg
    M_C = 12.0 * AMU    # 炭素原子の質量 (kg)
    K_B = 1.380649e-23  # ボルツマン定数 (J/K)
    
    def dV_to_T(dv):
        return 0.5 * M_C * (dv ** 2) / K_B
        
    def dV_err_to_T_err(dv, dv_err):
        return M_C * dv * dv_err / K_B

    # データa
    a_dV_err = dV_err_to_T_err(np.array(a_dV), np.array(a_dV_err))
    a_dV = dV_to_T(np.array(a_dV))
    
    # データb
    b_dV_err = dV_err_to_T_err(np.array(b_dV), np.array(b_dV_err))
    b_dV = dV_to_T(np.array(b_dV))
    b_dV_16 = dV_to_T(np.array(b_dV_16))
    b_dV_84 = dV_to_T(np.array(b_dV_84))
    
    # データc
    c_dV_err = dV_err_to_T_err(np.array(c_dV), np.array(c_dV_err))
    c_dV = dV_to_T(np.array(c_dV))
    
    # MCMC用データ
    if len(emcee_dV_16) > 0:
        emcee_dV_16 = dV_to_T(np.array(emcee_dV_16))
        emcee_dV_84 = dV_to_T(np.array(emcee_dV_84))
    if len(mc_dV_16) > 0:
        mc_dV_16 = dV_to_T(np.array(mc_dV_16))
        mc_dV_84 = dV_to_T(np.array(mc_dV_84))
        
    # 未補正データ
    if len(uncorr_R) > 0:
        uncorr_dV_err = dV_err_to_T_err(np.array(uncorr_dV), np.array(uncorr_dV_err))
        uncorr_dV = dV_to_T(np.array(uncorr_dV))

    # --- プロット ---
    plt.figure(figsize=(9, 6))
    
    # 全データを統合してRの昇順にソートする
    all_R = np.concatenate([a_R, b_R, c_R])
    all_dV = np.concatenate([a_dV, b_dV, c_dV])
    sort_idx = np.argsort(all_R)
    sorted_R = all_R[sort_idx]
    sorted_dV = all_dV[sort_idx]
    
    # Rが低い順に1本の線で繋ぐ
    plt.plot(sorted_R, sorted_dV, '-', color='black', linewidth=1.2, alpha=0.5, zorder=1, label='Connected Path')
    
    # 点の色は別々のままでプロット (エラーバー付き)
    plt.errorbar(a_R, a_dV, yerr=a_dV_err, fmt='o', color='orange', markersize=7, 
                 capsize=4, elinewidth=1.2, capthick=1.2, label='Data a', zorder=2)
    plt.errorbar(b_R, b_dV, yerr=b_dV_err, fmt='^', color='green', markersize=7, 
                 capsize=4, elinewidth=1.2, capthick=1.2, label='Data b', zorder=2)
    plt.errorbar(c_R, c_dV, yerr=c_dV_err, fmt='s', color='blue', markersize=7, 
                 capsize=4, elinewidth=1.2, capthick=1.2, label='Data c', zorder=2)
                 
    # 未補正データプロット
    if len(uncorr_R) > 0:
        plt.errorbar(uncorr_R, uncorr_dV, yerr=uncorr_dV_err, fmt='D-', color='gray', markersize=6, 
                     capsize=4, elinewidth=1.2, capthick=1.2, label='Uncorrected (Simple Fit)', zorder=1)
    
    # mcmc (a, b, c) の16%と84%を区別せずにつなげてプロット (塗りつぶし用)
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
        

        
        # 16%と84%の間を薄い色で塗りつぶす
        plt.fill_between(sorted_mc_R, sorted_mc_dV_16, sorted_mc_dV_84, color='purple', alpha=0.15, zorder=2, label='MCMC 16%-84% Range')
    
    # グラフの装飾
    plt.xlabel('Major Radius R (m)', fontsize=12)
    plt.ylabel('Temperature $T = m dV^2 / (2 k_B)$ (K)', fontsize=12)
    plt.title('Temperature Profile Comparison with Errors (a, b, c)', fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(fontsize=11)
    
    # Y軸の最小値を0に設定
    plt.ylim(bottom=0)
    
    # 温度が非常に大きくなる場合は科学的記数法を使う
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    plot_comparison()
