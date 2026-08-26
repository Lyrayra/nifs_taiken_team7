import numpy as np
import matplotlib.pyplot as plt

# 物理定数
C_MS = 299792458.0  # 光速 (m/s)
LAMBDA_0 = 529.05   # 静止波長 (nm)

def generate_velocity_distribution(v, A, v0, dV):
    """
    ガウス型速度分布関数 f(v) を計算する
    v : 速度のNumPy配列 (m/s)
    A : 面積（強度）の係数
    v0: 中心のバルク速度 (m/s)
    dV: 熱速度広がりパラメータ (m/s)
    """
    return (A / (np.sqrt(np.pi) * dV)) * np.exp(-((v - v0) / dV)**2)

def velocity_to_lambda(v, lambda0=LAMBDA_0):
    """ 速度 (m/s) から 波長 (nm) へのドップラー変換 """
    return lambda0 * (1.0 + v / C_MS)

def lambda_to_velocity(lam, lambda0=LAMBDA_0):
    """ 波長 (nm) から 速度 (m/s) への逆変換 (参考用) """
    return C_MS * (lam - lambda0) / lambda0

def generate_wavelength_distribution(lam, A, lambda0, v0, dV):
    """E(v) を E(λ) に変換した波長分布を計算する。"""
    v = lambda_to_velocity(lam, lambda0=lambda0)
    jacobian = C_MS / lambda0
    return generate_velocity_distribution(v, A, v0, dV) * jacobian

def load_average_spectrum(file_path):
    data_list = []

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

    frames = int(raw_data[:, 0].max()) + 1
    x_max = int(raw_data[:, 1].max()) + 1
    y_max = int(raw_data[:, 2].max()) + 1

    image_cube = np.zeros((frames, y_max, x_max))
    for row in raw_data:
        frame_idx, x_idx, y_idx, count = int(row[0]), int(row[1]), int(row[2]), row[3]
        image_cube[frame_idx, y_idx, x_idx] = count

    spectrum_data = np.mean(image_cube, axis=(0, 1))
    
    # 波長への変換を右側の1つ目のピークで位置合わせし、1pxあたりの分散を固定する
    peak1_idx = 80 + np.argmax(spectrum_data[80:105])
    
    # 指定された1pxあたりの波長 (値が減る方向なのでマイナス)
    dispersion = -0.029855
    
    # peak1_idx の波長を 528.03 nm として波長軸を計算
    x_indices = np.arange(x_max)
    wavelength_axis = 528.03 + (x_indices - peak1_idx) * dispersion
    baseline_samples = np.concatenate([spectrum_data[:5], spectrum_data[-5:]])
    baseline = float(np.mean(baseline_samples))
    low_outlier_cutoff = baseline - (float(np.max(spectrum_data)) - baseline) * 0.05
    masked_spectrum_data = spectrum_data.copy()
    masked_spectrum_data[masked_spectrum_data < low_outlier_cutoff] = np.nan

    return wavelength_axis, masked_spectrum_data, baseline

def plot_overlaid_spectrum_and_distribution(
    file_path,
    lambda0=LAMBDA_0,
    A_param=1E4,
    v0_param=15000.0,
    dV_param=300000.0,
    intensity_offset=2100,
):
    wavelength_axis, measured_spectrum, baseline = load_average_spectrum(file_path)
    v_offset = baseline

    theory = generate_wavelength_distribution(wavelength_axis, A_param, lambda0, v0_param, dV_param)
    theory = theory + intensity_offset

    plt.figure(figsize=(10, 5))
    plt.plot(wavelength_axis, measured_spectrum, color='blue', label='takaave spectrum')
    plt.plot(wavelength_axis, theory, color='red', linestyle='--', label='E(λ) converted from E(v)')
    plt.axhline(v_offset, color='gray', linestyle=':', alpha=0.8, label='v_offset')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Count (cnt)')
    plt.title('Measured Spectrum and Converted E(λ) Overlay')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_overlaid_spectrum_and_distribution('a260825_img.txt')