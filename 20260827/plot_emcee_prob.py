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
            next(reader)
            for row in reader:
                if not row: continue
                a_R.append(float(row[1]))
                a_dV.append(float(row[3]))
                if len(row) > 4:
                    a_dV_err.append(float(row[4]))
                else:
                    a_dV_err.append(0.0)
    except FileNotFoundError:
        print(f"エラー: {csv_file} が見つかりません。")
        return
            
    # --- bのデータ (画像から書き起こしたもの + 誤差) ---
    b_data = np.array([
        [0, 3.62962, 192439.9877939988,  11925.03505374918],
        [1, 3.6939,  245616.06530712068, 10457.351299181802],
        [2, 3.75868, 248914.7592786996,  10025.693074786484],
        [3, 3.83491, 233249.58298758548, 5966.759459465759],
        [4, 3.93396, 203064.91313932854, 4063.2225913846432],
        [5, 4.0342,  203676.21002683835, 2804.9898767972054],
        [6, 4.13564, 172496.9011050146,  2249.8181173319213],
        [7, 4.23832, 164262.26199249073, 1364.0632902031],
        [8, 4.34225, 136953.750982935,   1094.4103715900956],
        [9, 4.4357,  126657.08477723994, 930.4047534014084],
        [10, 4.50646, 108573.06372809905, 870.8415709304451],
        [11, 4.5778,  95828.82416929209,  1644.6870257624037]
    ])
    b_R = b_data[:, 1]
    b_dV = b_data[:, 2]
    b_dV_err = b_data[:, 3]
    
    # --- cのデータ (data_n.csv) ---
    c_R = []
    c_dV = []
    c_dV_err = []
    try:
        data_n_file = os.path.join(SCRIPT_DIR, 'data_n.csv')
        with open(data_n_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if not row: continue
                c_R.append(float(row[0]))
                c_dV.append(float(row[1]))
                if len(row) > 2:
                    c_dV_err.append(float(row[2]))
                else:
                    c_dV_err.append(0.0)
    except FileNotFoundError:
        print(f"エラー: data_n.csv が見つかりません。")
        return

    # --- プロット ---
    plt.figure(figsize=(10, 7))

    # ==========================================================
    # emceeの事後分布を確率分布（色の濃淡）としてプロット
    # ==========================================================
    npz_file = os.path.join(SCRIPT_DIR, 'emcee_samples.npz')
    if os.path.exists(npz_file):
        npz_data = np.load(npz_file)
        
        # 1〜12チャンネルのデータを取得
        channels = range(1, 13)
        R_list = []
        samples_list = []
        for ch in channels:
            key_r = f"ch_{ch}_R"
            key_s = f"ch_{ch}_dV_samples"
            if key_r in npz_data and key_s in npz_data:
                R_list.append(npz_data[key_r].item())
                samples_list.append(npz_data[key_s])
        
        # 1. バイオリンプロットを使って幅で確率分布を表現
        parts = plt.violinplot(samples_list, positions=R_list, widths=0.03, showextrema=False)
        for pc in parts['bodies']:
            pc.set_facecolor('purple')
            pc.set_edgecolor('none')
            pc.set_alpha(0.3)
            pc.set_zorder(1)
        
        # 2. 点の散布（散布図）で色の濃淡を表現
        # alphaを極端に下げることで、点が密集している＝確率が高いところが濃くなる
        for R_val, samps in zip(R_list, samples_list):
            # 視覚的な幅を持たせるため、Rにランダムなジッター(ブレ)を加える
            jitter = np.random.normal(0, 0.003, size=len(samps))
            plt.scatter(R_val + jitter, samps, color='purple', alpha=0.015, s=2, zorder=2)
            
        # 凡例用のダミープロット
        plt.plot([], [], color='purple', alpha=0.6, linewidth=8, label='emcee Posterior Density')
    else:
        print(f"警告: {npz_file} が見つかりません。emceeの分布表示をスキップします。")


    # ==========================================================
    # 既存の棒グラフ (エラーバー) を重ねる
    # ==========================================================
    
    # データを統合してRの昇順にソート (線で繋ぐため)
    all_R = np.concatenate([a_R, b_R, c_R])
    all_dV = np.concatenate([a_dV, b_dV, c_dV])
    sort_idx = np.argsort(all_R)
    sorted_R = all_R[sort_idx]
    sorted_dV = all_dV[sort_idx]
    
    plt.plot(sorted_R, sorted_dV, '-', color='black', linewidth=1.2, alpha=0.5, zorder=3, label='Connected Path')
    
    plt.errorbar(a_R, a_dV, yerr=a_dV_err, fmt='o', color='orange', markersize=7, 
                 capsize=4, elinewidth=1.2, capthick=1.2, label='Data a (gyaku.py/gosa.py)', zorder=4)
    plt.errorbar(b_R, b_dV, yerr=b_dV_err, fmt='^', color='green', markersize=7, 
                 capsize=4, elinewidth=1.2, capthick=1.2, label='Data b (Provided)', zorder=4)
    plt.errorbar(c_R, c_dV, yerr=c_dV_err, fmt='s', color='blue', markersize=7, 
                 capsize=4, elinewidth=1.2, capthick=1.2, label='Data c (data_n.csv)', zorder=4)
    
    # グラフの装飾
    plt.xlabel('Major Radius R (m)', fontsize=12)
    plt.ylabel('Thermal Width dV (m/s)', fontsize=12)
    plt.title('Thermal Width Profile: Point Estimates vs emcee Posterior Density', fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(fontsize=11, loc='best')
    
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    plot_comparison()
