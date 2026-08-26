import numpy as np
import matplotlib.pyplot as plt
import csv

def plot_comparison():
    # --- aのデータ (今回 gyaku.py で計算して保存したもの) を読み込む ---
    a_R = []
    a_dV = []
    
    csv_file = 'dv_r_profile.csv'
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader) # ヘッダーをスキップ
            for row in reader:
                if not row: continue
                # 1列目がR, 3列目がdV
                a_R.append(float(row[1]))
                a_dV.append(float(row[3])) # m/s のまま
    except FileNotFoundError:
        print(f"エラー: {csv_file} が見つかりません。先に gyaku.py を実行してください。")
        return
            
    # --- bのデータ (画像から書き起こしたもの) ---
    b_data = np.array([
        [3.62962, 187423.15441520754],
        [3.6939,  241480.98887838708],
        [3.75868, 244608.14749254324],
        [3.83491, 228816.07256185322],
        [3.93396, 198478.26802721768],
        [4.0342,  197981.6634977722],
        [4.13564, 166667.63895776126],
        [4.23832, 156868.84561887136],
        [4.34225, 129202.4202137534],
        [4.4357,  117716.25251674395],
        [4.50646, 99864.58191630078],
        [4.5778,  86217.6026472956]
    ])
    b_R = b_data[:, 0]
    b_dV = b_data[:, 1]
    
    # --- cのデータ (data_n.csv) ---
    c_R = []
    c_dV = []
    try:
        with open('data_n.csv', 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader) # ヘッダーをスキップ
            for row in reader:
                if not row: continue
                c_R.append(float(row[0]))
                c_dV.append(float(row[1])) # m/s のまま
        c_R = np.array(c_R)
        c_dV = np.array(c_dV)
    except FileNotFoundError:
        print(f"エラー: data_n.csv が見つかりません。")
        return
    
    # --- プロット ---
    plt.figure(figsize=(8, 6))
    
    # 全データを統合してRの昇順にソートする
    all_R = np.concatenate([a_R, b_R, c_R])
    all_dV = np.concatenate([a_dV, b_dV, c_dV])
    sort_idx = np.argsort(all_R)
    sorted_R = all_R[sort_idx]
    sorted_dV = all_dV[sort_idx]
    
    # Rが低い順に1本の線で繋ぐ
    plt.plot(sorted_R, sorted_dV, '-', color='black', linewidth=1.5, zorder=1, label='Connected Path')
    
    # 点の色は別々のままでプロット (線なし)
    plt.plot(a_R, a_dV, 'o', color='orange', markersize=8, label='Data a (gyaku.py)', zorder=2)
    plt.plot(b_R, b_dV, '^', color='green', markersize=8, label='Data b (Provided)', zorder=2)
    plt.plot(c_R, c_dV, 's', color='blue', markersize=8, label='Data c (Provided)', zorder=2)
    
    # グラフの装飾
    plt.xlabel('Major Radius R (m)', fontsize=12)
    plt.ylabel('Thermal Width dV (m/s)', fontsize=12)
    plt.title('Thermal Width Profile Comparison (a, b, c)', fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(fontsize=12)
    
    # Y軸の最小値を0に設定
    plt.ylim(bottom=0)
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    plot_comparison()
