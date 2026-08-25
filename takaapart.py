import numpy as np
import matplotlib.pyplot as plt


def load_image_cube(file_path):
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

    return image_cube, x_max, y_max


def calibrate_wavelength_axis(x_max):
    pixel_peaks = np.array([12, 33, 92, 111])
    wavelength_peaks = np.array([530.47580, 529.81891, 528.03, 527.40393])
    poly = np.poly1d(np.polyfit(pixel_peaks, wavelength_peaks, 1))
    wavelength_axis = np.linspace(poly(0), poly(x_max - 1), x_max)
    return wavelength_axis, poly, wavelength_peaks


def plot_four_wavelength_windows(file_path, target_frame=0, window_nm=0.45):
    image_cube, x_max, y_max = load_image_cube(file_path)
    wavelength_axis, _, wavelength_peaks = calibrate_wavelength_axis(x_max)
    wavelength_peaks = np.array([529.81891, 530.47580])
    low_outlier_cutoff = -0.05

    spectrum = np.mean(image_cube, axis=(0, 1))
    x_windows = []
    y_windows = []
    normalized_windows = []
    baseline_values = []

    for center_wavelength in wavelength_peaks:
        left_pad = window_nm
        right_pad = window_nm
        if center_wavelength == np.min(wavelength_peaks):
            left_pad = window_nm * 1.4

        mask = (wavelength_axis >= center_wavelength - left_pad) & (wavelength_axis <= center_wavelength + right_pad)
        x_window = wavelength_axis[mask]
        y_window = spectrum[mask]

        x_windows.append(x_window)
        y_windows.append(y_window)

        if x_window.size == 0:
            baseline_values.append(None)
            normalized_windows.append(y_window)
        else:
            edge_count = max(1, min(5, x_window.size // 5))
            baseline_samples = np.concatenate([y_window[:edge_count], y_window[-edge_count:]])
            baseline = float(np.mean(baseline_samples))
            baseline_values.append(baseline)
            peak = float(np.max(y_window))
            scale = peak - baseline if peak > baseline else 1.0
            normalized_window = (y_window - baseline) / scale
            normalized_window[normalized_window < low_outlier_cutoff] = np.nan
            normalized_windows.append(normalized_window)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), sharey=True)
    axes = axes.ravel()

    for idx, center_wavelength in enumerate(wavelength_peaks):
        ax = axes[idx]
        x_window = x_windows[idx]
        y_window = normalized_windows[idx]

        if x_window.size == 0:
            ax.text(0.5, 0.5, 'No data in window', ha='center', va='center', transform=ax.transAxes)
        else:
            ax.plot(x_window, y_window, color='blue')

            baseline = baseline_values[idx]
            ax.axhline(0.0, color='gray', linestyle=':', alpha=0.8, label='Baseline')
            ax.axvline(center_wavelength, color='red', linestyle='--', alpha=0.7)
            left_pad = window_nm
            right_pad = window_nm
            if center_wavelength == np.min(wavelength_peaks):
                left_pad = window_nm * 1.4
            ax.set_xlim(center_wavelength - left_pad, center_wavelength + right_pad)
            ax.set_ylim(-0.15, 1.15)

        ax.set_title(f'Window around {center_wavelength:.5f} nm')
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Normalized Count')
        ax.grid(True, linestyle='--', alpha=0.4)

    fig.suptitle('529.81891 nm and 530.47580 nm Windows from the Average Spectrum', y=1.02)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_four_wavelength_windows('a260825_img.txt', target_frame=0)
