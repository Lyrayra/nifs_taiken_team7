import numpy as np
import matplotlib.pyplot as plt

def plot_overlapping_frames(file_path):
    data_list = []
    
    # 1. データの読み込み
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
                    data_list.append([float(v) for v in parts])

    raw_data = np.array(data_list)

    if raw_data.size == 0:
        raise ValueError(f'No data rows were found in {file_path}. Check the [data] section marker.')
    
    # 2. 3次元配列 (frame, y, x) の構築
    frames = int(raw_data[:, 0].max()) + 1
    x_max = int(raw_data[:, 1].max()) + 1
    y_max = int(raw_data[:, 2].max()) + 1
    
    image_cube = np.zeros((frames, y_max, x_max))
    for row in raw_data:
        f, x, y, count = int(row[0]), int(row[1]), int(row[2]), row[3]
        image_cube[f, y, x] = count

    # 3. 波長キャリブレーション（taka.py と同じ補正）
    pixel_peaks = np.array([12, 33, 92, 111])
    wavelength_peaks = np.array([530.47580, 529.81891, 528.03, 527.40393])
    poly = np.poly1d(np.polyfit(pixel_peaks, wavelength_peaks, 1))
    wl_start = poly(0)
    wl_end = poly(x_max - 1)

    # 4. アプローチA: 平均ヒートマップ (全フレームの平均)
    accumulated_heatmap = np.mean(image_cube, axis=0)
    accumulated_heatmap_mirrored = np.fliplr(accumulated_heatmap)
    
    plt.figure(figsize=(10, 4))
    plt.imshow(accumulated_heatmap_mirrored, cmap='viridis', aspect='auto', origin='lower',
               extent=[wl_end, wl_start, 0, y_max - 1])
    plt.colorbar(label='Average Count (cnt)')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Spatial y (pixel)')
    plt.title('Approach A: Average Heatmap (Wavelength Calibrated)')
    plt.tight_layout()
    plt.show()

    # 5. アプローチB: 全フレーム平均の折れ線グラフ
    spectrum_data = np.mean(image_cube, axis=(0, 1))
    wavelength_axis = np.linspace(wl_start, wl_end, x_max)
    
    plt.figure(figsize=(10, 5))
    plt.plot(wavelength_axis, spectrum_data, color='blue')

    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Average Count (cnt) [Averaged over y and frames]')
    plt.title('Approach B: Average Line Plot of All Frames (Wavelength Calibrated)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 対象ファイル名を指定して実行
    plot_overlapping_frames('a260825_img.txt')