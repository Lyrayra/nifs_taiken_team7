import numpy as np
import matplotlib.pyplot as plt

def plot_calibrated_heatmap(file_path, target_frame=0):
    data_list = []
    
    # 1. データの読み込みとパース
    with open(file_path, 'r') as f:
        is_data_section = False
        for line in f:
            line = line.strip()
            if line == '[data]' or line == '#[data]':
                is_data_section = True
                continue
            if is_data_section and line:
                parts = line.split(',')
                if len(parts) == 4:
                    data_list.append([int(parts[0]), int(parts[1]), int(parts[2]), float(parts[3])])

    # 2. NumPy配列への変換とデータ存在チェック
    data = np.array(data_list)
    if data.size == 0:
        raise ValueError(f'No data rows were found in {file_path}. Check the [data] section marker.')
    
    # 指定フレームのデータを抽出
    frame_data = data[data[:, 0] == target_frame]
    
    # グリッドサイズの決定
    x_max = int(data[:, 1].max()) + 1
    y_max = int(data[:, 2].max()) + 1
    
    # ヒートマップの生成
    heatmap = np.zeros((y_max, x_max))
    for row in frame_data:
        heatmap[int(row[2]), int(row[1])] = row[3]

    # 3. 波長キャリブレーション（線形フィッティング）
    pixel_peaks = np.array([12, 33, 92, 111])
    wavelength_peaks = np.array([530.47580, 529.81891, 528.03, 527.40393])
    
    # 1次関数で近似
    poly = np.poly1d(np.polyfit(pixel_peaks, wavelength_peaks, 1))
    
    # 元の画像の左端と右端の波長を算出
    wl_start = poly(0)
    wl_end = poly(x_max - 1)

    # ★追加・変更部分: 画像を左右反転(鏡写し)する
    heatmap_mirrored = np.fliplr(heatmap)

    # 4. 描画処理
    plt.figure(figsize=(10, 4))
    
    # グラフの描画 (extentを小さい波長 wl_end から大きい波長 wl_start の順に設定)
    plt.imshow(heatmap_mirrored, cmap='viridis', aspect='auto', origin='lower',
               extent=[wl_end, wl_start, 0, y_max - 1])
    
    plt.colorbar(label='Count (cnt)')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Spatial y (pixel)')
    plt.title(f'Spectral Heatmap')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 対象ファイル名とプロットしたいフレーム番号を指定
    plot_calibrated_heatmap('a260825_img.txt', target_frame=0)