import os

import numpy as np
import matplotlib.pyplot as plt

from ilambda_builder import make_i_lambda
from sokudobunpu import LAMBDA_0, generate_wavelength_distribution


def load_i_lambda(file_path='I_lambda.npz', source_file='a260825_img.txt'):
    if os.path.exists(file_path):
        data = np.load(file_path)
        wavelength_axis = data['wavelength_axis']
        i_lambda = data['I_lambda']
        if wavelength_axis[0] > wavelength_axis[-1]:
            wavelength_axis = wavelength_axis[::-1]
            i_lambda = i_lambda[::-1]
        return wavelength_axis, i_lambda

    wavelength_axis, i_lambda, _ = make_i_lambda(source_file)
    if wavelength_axis[0] > wavelength_axis[-1]:
        wavelength_axis = wavelength_axis[::-1]
        i_lambda = i_lambda[::-1]
    np.savez(file_path, wavelength_axis=wavelength_axis, I_lambda=i_lambda)
    return wavelength_axis, i_lambda


def normalize_area(values):
    total = float(np.sum(values))
    if total <= 0:
        raise ValueError('Distribution area became non-positive.')
    return values / total


def center_to_peak(wavelength_axis, values):
    peak_index = int(np.argmax(values))
    center_index = values.size // 2
    shift = center_index - peak_index

    left_pad = max(0, 2 * shift)
    right_pad = max(0, -2 * shift)
    centered_values = np.pad(values, (left_pad, right_pad), mode='constant')

    delta_lambda = abs(float(np.mean(np.diff(wavelength_axis))))
    axis_start = wavelength_axis[0] - left_pad * delta_lambda
    centered_axis = axis_start + np.arange(centered_values.size) * delta_lambda

    return centered_axis, centered_values, {
        'peak_index': peak_index,
        'center_index': center_index,
        'shift': shift,
        'left_pad': left_pad,
        'right_pad': right_pad,
    }


def pad_to_length(values, wavelength_axis, target_length):
    if values.size == target_length:
        return values, wavelength_axis

    pad_total = target_length - values.size
    if pad_total < 0:
        raise ValueError('target_length must be greater than or equal to the input length.')

    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    padded_values = np.pad(values, (pad_left, pad_right), mode='constant')

    delta_lambda = abs(float(np.mean(np.diff(wavelength_axis))))
    axis_start = wavelength_axis[0] - pad_left * delta_lambda
    padded_axis = axis_start + np.arange(target_length) * delta_lambda
    return padded_values, padded_axis


def build_e_lambda(wavelength_axis, lambda0=LAMBDA_0, A_param=5e3, v0_param=15000.0, dV_param=300000.0):
    e_lambda = generate_wavelength_distribution(
        wavelength_axis,
        A_param,
        lambda0,
        v0_param,
        dV_param,
    )
    return normalize_area(e_lambda)


def convolve_distributions(e_lambda, i_lambda, delta_lambda):
    return np.convolve(e_lambda, i_lambda, mode='same')


def plot_convolution(
    source_file='a260825_img.txt',
    i_lambda_file='I_lambda.npz',
    lambda0=LAMBDA_0,
    A_param=5e3,
    v0_param=15000.0,
    dV_param=300000.0,
):
    wavelength_axis, i_lambda = load_i_lambda(i_lambda_file, source_file=source_file)
    e_lambda = build_e_lambda(wavelength_axis, lambda0=lambda0, A_param=A_param, v0_param=v0_param, dV_param=dV_param)

    centered_wavelength_axis, centered_i_lambda, i_info = center_to_peak(wavelength_axis, normalize_area(i_lambda))
    centered_wavelength_axis_e, centered_e_lambda, e_info = center_to_peak(wavelength_axis, e_lambda)

    target_length = max(centered_e_lambda.size, centered_i_lambda.size)
    centered_e_lambda, centered_wavelength_axis_e = pad_to_length(centered_e_lambda, centered_wavelength_axis_e, target_length)
    centered_i_lambda, centered_wavelength_axis = pad_to_length(centered_i_lambda, centered_wavelength_axis, target_length)

    delta_lambda = abs(float(np.mean(np.diff(centered_wavelength_axis))))
    convolution = convolve_distributions(centered_e_lambda, centered_i_lambda, delta_lambda)
    convolution = normalize_area(convolution)

    plt.figure(figsize=(11, 5))
    plt.plot(centered_wavelength_axis_e, centered_e_lambda, color='red', linestyle='--', label='E(λ)')
    plt.plot(centered_wavelength_axis, centered_i_lambda, color='blue', label='I(λ)')
    plt.plot(centered_wavelength_axis, convolution, color='black', linewidth=2.0, label='E(λ) * I(λ)')
    plt.axvline(centered_wavelength_axis[len(centered_wavelength_axis) // 2], color='gray', linestyle=':', alpha=0.8)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Normalized Intensity')
    plt.title('Convolution of E(λ) and I(λ)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

    np.savez(
        'convolution_result.npz',
        wavelength_axis=centered_wavelength_axis,
        E_lambda=centered_e_lambda,
        I_lambda=centered_i_lambda,
        convolution=convolution,
        delta_lambda=delta_lambda,
        E_peak_index=e_info['peak_index'],
        I_peak_index=i_info['peak_index'],
        E_shift=e_info['shift'],
        I_shift=i_info['shift'],
    )
    print('Saved: convolution_result.npz')
    print(f'delta_lambda = {delta_lambda:.8f} nm')
    print(f"E shift = {e_info['shift']}, I shift = {i_info['shift']}")


if __name__ == '__main__':
    plot_convolution()
