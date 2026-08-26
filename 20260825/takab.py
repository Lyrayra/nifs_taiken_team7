import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import shift

# データのパラメータ設定
num_frames = 50
size_x = 128
size_y = 12

# データを格納する3次元配列を準備
data = np.zeros((num_frames, size_x, size_y))

# データの読み込み
file_path = 'b260825_img.txt'

with open(file_path, 'r') as f:
    for line in f:
        if line.startswith('#') or not line.strip():
            continue
        
        parts = line.strip().split(',')
        if len(parts) == 4:
            f_idx = int(parts[0])
            x_idx = int(parts[1])
            y_idx = int(parts[2])
            count = float(parts[3])
            
            data[f_idx, x_idx, y_idx] = count

# フレーム0のデータを抽出
frame_idx = 0
img = data[frame_idx, :, :]  # 形状: (128, 128)

# ==========================================
# 改善版のピーク補正処理（直線近似）
# ==========================================

# 1. y軸方向ごとの最大カウント値を計算し、信号がある行（y）を特定する
row_max = np.max(img, axis=0)
threshold = np.max(row_max) * 0.5  # 全体の最大値の50%を閾値とする
valid_y = np.where(row_max > threshold)[0]

# 2. 信号が強い行だけで、x方向のピーク位置を取得する
valid_peaks_x = np.argmax(img[:, valid_y], axis=0)

# 3. 取得したピーク位置から、直線の傾きを近似する（1次関数: x = a*y + b）
coefficients = np.polyfit(valid_y, valid_peaks_x, 1)
poly_func = np.poly1d(coefficients)

# 4. 近似した直線を使って、すべての y に対する理想的なピーク位置を計算する
ideal_peaks_x = poly_func(np.arange(size_y))

# 5. 基準となる x 座標を決定（ここでは信号領域の最初の行のピーク位置）
reference_peak = ideal_peaks_x[valid_y[0]]

# 6. 補正後の画像を格納する配列
corrected_img = np.zeros_like(img)

# 各 y について、理想的なピーク位置からのズレを計算してシフトする
for y in range(size_y):
    # 【変更点1】四捨五入を外し、小数のままシフト量を保持する
    shift_amount = reference_peak - ideal_peaks_x[y]
    
    # 【変更点2】np.roll の代わりにサブピクセル補間でシフトする
    # ※ mode='wrap' で np.roll と同じように端をループさせます
    corrected_img[:, y] = shift(img[:, y], shift_amount, mode='wrap')

# ==========================================
# プロット
# ==========================================
plt.figure(figsize=(12, 5))

# 補正前の画像
plt.subplot(1, 2, 1)
plt.title(f"Original (Frame {frame_idx})")
plt.imshow(img.T, origin='lower', aspect='auto', cmap='viridis')
# 信頼できるピーク（赤点）と、画像全体を貫く近似直線（オレンジ点線）
plt.scatter(valid_peaks_x, valid_y, color='red', s=15, label='Valid Peaks')
plt.plot(ideal_peaks_x, np.arange(size_y), color='orange', linestyle='--', label='Fitted Line')
plt.xlabel("x (Wavelength)")
plt.ylabel("y (Space)")
plt.legend()

# 補正後の画像
plt.subplot(1, 2, 2)
plt.title(f"Corrected (Frame {frame_idx})")
plt.imshow(corrected_img.T, origin='lower', aspect='auto', cmap='viridis')
plt.axvline(x=reference_peak, color='orange', linestyle='--', label='Aligned Line')
plt.xlabel("x (Wavelength)")
plt.ylabel("y (Space)")
plt.legend()

plt.tight_layout()
plt.show()