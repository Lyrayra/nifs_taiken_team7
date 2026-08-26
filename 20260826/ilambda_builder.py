import numpy as np
import matplotlib.pyplot as plt


def load_spectra_and_wavelengths(file_path):
    """
    ファイルから画像キューブを読み込み、チャンネルごとのスペクトルと波長軸（2D）を計算して返す。
    波長軸のキャリブレーションは、各チャンネルごとに右側の2つのピークを用いて行う。
    """
    data_list = []
    with open(file_path, 'r') as f:
        is_data = False
        for line in f:
            line = line.strip()
            if line == '[data]' or line == '#[data]':
                is_data = True
                continue
            if is_data and line:
                parts = line.split(',')
                if len(parts) == 4:
                    data_list.append([float(v) for v in parts])
    
    raw_data = np.array(data_list)
    frames = int(raw_data[:, 0].max()) + 1
    x_max = int(raw_data[:, 1].max()) + 1
    y_max = int(raw_data[:, 2].max()) + 1
    
    image_cube = np.zeros((frames, y_max, x_max))
    for row in raw_data:
        frame_idx, x_idx, y_idx, count = int(row[0]), int(row[1]), int(row[2]), row[3]
        image_cube[frame_idx, y_idx, x_idx] = count

    # 時間平均をとって (12, 128) の2Dスペクトルにする
    img = np.mean(image_cube, axis=0)
    
    # チャンネルごとの波長軸を格納する配列 (128, 12)
    wavelength_axis = np.zeros((x_max, y_max))
    
    wne1 = 529.81891
    wne2 = 530.47573
    pixel = np.arange(x_max)
    
    from scipy.signal import find_peaks
    
    for i in range(y_max):
        spct = img[i, :]
        # ピークを検出
        peaks, _ = find_peaks(spct)
        
        if len(peaks) < 2:
            print(f"Warning: Not enough peaks found in channel {i}")
            continue
            
        # ピークの強度順にソート
        top_peak_order = np.argsort(spct[peaks])[::-1]
        
        # 上位2つのピークのインデックスを取得
        idx1 = peaks[top_peak_order[0]]
        idx2 = peaks[top_peak_order[1]]
        
        # 波長を線形補間して計算
        wavelength_axis[:, i] = (pixel - idx1) / (idx2 - idx1) * (wne2 - wne1) + wne1
        
    return wavelength_axis, img.T  # img.T makes it (128, 12)


def estimate_baseline(spectrum, edge_count=5):
    edge_count = min(edge_count, spectrum.size // 2)
    edge_samples = np.concatenate([spectrum[:edge_count], spectrum[-edge_count:]])
    return float(np.mean(edge_samples))



def make_i_lambda(file_path, edge_count=5, target_center_nm=None, window_nm=0.35, channel=None):
    wavelength_axis_2d, spectra_2d = load_spectra_and_wavelengths(file_path)
    
    if channel is not None:
        wavelength_axis = wavelength_axis_2d[:, channel]
        spectrum = spectra_2d[:, channel]
    else:
        wavelength_axis = wavelength_axis_2d[:, 0]
        spectrum = spectra_2d[:, 0]

    # ターゲット波長が指定されていない場合は、スペクトルの最大値をピークとみなす
    if target_center_nm is None:
        peak_idx = np.argmax(spectrum)
        target_center_nm = wavelength_axis[peak_idx]

    # ピークの周辺一山分だけを切り出す（マスクの作成）
    mask = (wavelength_axis >= target_center_nm - window_nm) & (wavelength_axis <= target_center_nm + window_nm)
    wavelength_window = wavelength_axis[mask]
    spectrum_window = spectrum[mask]

    if spectrum_window.size == 0:
        raise ValueError(f"No data found around the target peak {target_center_nm} nm within ±{window_nm} nm.")

    # 1) バックグラウンド除去（切り出したウィンドウの端のデータからベースラインを推定）
    baseline = estimate_baseline(spectrum_window, edge_count=edge_count)
    spectrum_bg_subtracted = spectrum_window - baseline
    spectrum_bg_subtracted[spectrum_bg_subtracted < 0] = 0.0

    # 2) 面積正規化
    area = float(np.sum(spectrum_bg_subtracted))
    if area <= 0:
        raise ValueError('Spectrum area became non-positive after baseline subtraction.')
    i_lambda = spectrum_bg_subtracted / area

    # 3) ピーク中心化（ピークを配列中央へ移動）
    peak_index = int(np.argmax(i_lambda))
    center_index = i_lambda.size // 2
    shift = center_index - peak_index

    centered_i_lambda = np.zeros_like(i_lambda)
    if shift >= 0:
        src_start = 0
        src_end = i_lambda.size - shift
        dst_start = shift
        dst_end = shift + (src_end - src_start)
    else:
        src_start = -shift
        src_end = i_lambda.size
        dst_start = 0
        dst_end = src_end - src_start

    centered_i_lambda[dst_start:dst_end] = i_lambda[src_start:src_end]

    centered_wavelength_axis = wavelength_window - wavelength_window[peak_index] + wavelength_window[center_index]

    return centered_wavelength_axis, centered_i_lambda, {
        'baseline': baseline,
        'peak_index': peak_index,
        'center_index': center_index,
        'shift': shift,
        'area': area,
    }


def plot_i_lambda(file_path='a260825_img.txt', edge_count=5):
    wavelength_axis, i_lambda, info = make_i_lambda(file_path, edge_count=edge_count)

    plt.figure(figsize=(10, 5))
    plt.plot(wavelength_axis, i_lambda, color='blue')
    plt.axvline(wavelength_axis[len(wavelength_axis) // 2], color='red', linestyle='--', alpha=0.7, label='center')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('I(λ)')
    plt.title('I(λ): baseline-subtracted, normalized, centered')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

    np.savez('I_lambda.npz', wavelength_axis=wavelength_axis, I_lambda=i_lambda, **info)
    print('Saved: I_lambda.npz')
    print(f"baseline = {info['baseline']:.6f}, area = {info['area']:.6f}, peak_index = {info['peak_index']}, shift = {info['shift']}")


if __name__ == '__main__':
    plot_i_lambda()
