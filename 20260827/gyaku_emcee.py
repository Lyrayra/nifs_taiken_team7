"""
emcee を使った MCMC による逆問題解析。
gyaku.py の curve_fit の代わりに emcee でパラメータの事後分布をサンプリングし、
dV の分布を各チャンネルごとに可視化する。
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
from scipy.optimize import curve_fit
import emcee
import corner

# 他のスクリプトからのインポート
from sokudobunpu import generate_wavelength_distribution
from ilambda_builder import make_i_lambda
from forward_model import load_dat_spectrum


def make_model(i_lambda_centered, target_line, full_wl_axis):
    """
    モデル関数を生成するファクトリ関数 (gyaku.py と同一)。
    """
    def model_func(x_subset, A, v0, dV, bg):
        e_lambda = generate_wavelength_distribution(full_wl_axis, A=A, lambda0=target_line, v0=v0, dV=dV)
        d_lambda_full = convolve(e_lambda, i_lambda_centered, mode='same')
        d_lambda_total = d_lambda_full + bg
        if full_wl_axis[0] > full_wl_axis[-1]:
            return np.interp(x_subset, full_wl_axis[::-1], d_lambda_total[::-1])
        else:
            return np.interp(x_subset, full_wl_axis, d_lambda_total)
    return model_func


def log_prior(theta):
    """
    一様事前分布 (flat prior)。
    パラメータが物理的に妥当な範囲にあるかチェック。
    """
    A, v0, dV, bg = theta
    if A < 0:
        return -np.inf
    if dV < 1000 or dV > 1e7:
        return -np.inf
    if abs(v0) > 1e6:
        return -np.inf
    return 0.0


def log_likelihood(theta, x, y, model_func, sigma):
    """
    ガウス型の尤度関数。
    """
    A, v0, dV, bg = theta
    try:
        y_model = model_func(x, A, v0, dV, bg)
    except Exception:
        return -np.inf
    residual = y - y_model
    return -0.5 * np.sum((residual / sigma) ** 2)


def log_probability(theta, x, y, model_func, sigma):
    """
    事後確率 = 事前分布 × 尤度。
    """
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(theta, x, y, model_func, sigma)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll


def solve_with_emcee():
    # 対象のピーク波長
    inst_peak = 529.81891
    target_line = 529.81891

    # 観測データの読み込み
    dat_file = 'lhdcxs9a_img_sig@189129_t4.44s.txt'
    dat_wl, dat_spectra = load_dat_spectrum(dat_file)
    num_channels = dat_spectra.shape[1]

    # prep ファイルを読み込んでチャンネル→主半径 R の対応を取得
    from reff import load_prep
    prep_file = 'lhdcxs9a_prep@189129.dat'
    ch_arr, r_obs_arr = load_prep(prep_file)
    ch_to_r = {int(ch): r for ch, r in zip(ch_arr, r_obs_arr)}

    # MCMC パラメータ
    ndim = 4           # パラメータ数: A, v0, dV, bg
    nwalkers = 32      # ウォーカー数
    nsteps = 2000      # ステップ数
    burn_in = 500      # バーンイン

    # 結果保存
    results = []

    # --- 図1: 各チャンネルの dV 事後分布ヒストグラム ---
    fig_hist, axes_hist = plt.subplots(3, 4, figsize=(16, 10))
    axes_hist = axes_hist.flatten()

    print("【emcee MCMC 解析】")
    print(f"  Walkers: {nwalkers}, Steps: {nsteps}, Burn-in: {burn_in}")
    print(f"{'Ch':>3} | {'R (m)':>8} | {'dV_median':>12} | {'dV_16%':>12} | {'dV_84%':>12} | {'σ_dV':>10}")
    print("-" * 70)

    for ch in range(num_channels):
        y = dat_spectra[:, ch]
        wl = dat_wl[:, ch]

        # 装置関数
        _, i_lambda, _ = make_i_lambda('a260825_img.txt', target_center_nm=inst_peak, window_nm=0.35, channel=ch)
        peak_idx = np.argmax(i_lambda)
        shift = len(i_lambda) // 2 - peak_idx
        i_lambda_centered = np.roll(i_lambda, shift)
        i_lambda_centered = i_lambda_centered / np.sum(i_lambda_centered)

        # モデル関数
        model = make_model(i_lambda_centered, target_line, wl)

        # NaN 除去
        mask = ~np.isnan(y)
        x_fit = wl[mask]
        y_fit = y[mask]

        # まず curve_fit で初期値を求める
        A_guess = (np.max(y_fit) - np.min(y_fit)) * 1000
        bg_guess = np.min(y_fit)
        p0 = [A_guess, 0.0, 50000.0, bg_guess]
        bounds = ([0, -1e6, 1000, -np.inf], [np.inf, 1e6, 1e7, np.inf])

        try:
            popt, _ = curve_fit(model, x_fit, y_fit, p0=p0, bounds=bounds)
        except Exception as e:
            print(f"Ch {ch+1} curve_fit failed: {e}")
            axes_hist[ch].set_title(f'Ch {ch+1}: Failed')
            axes_hist[ch].text(0.5, 0.5, 'Fit Failed', ha='center', va='center', transform=axes_hist[ch].transAxes)
            results.append({'ch': ch + 1, 'error': True})
            continue

        # データのノイズレベルを残差から推定
        y_pred_init = model(x_fit, *popt)
        sigma = np.std(y_fit - y_pred_init)
        if sigma < 1e-15:
            sigma = 1e-6  # ゼロ除算防止

        # 初期ウォーカー位置: curve_fit の最適値周辺に散らばらせる
        pos = popt + 1e-4 * np.abs(popt) * np.random.randn(nwalkers, ndim)
        # A が負にならないようにクリップ
        pos[:, 0] = np.abs(pos[:, 0])
        # dV が 1000 未満にならないようにクリップ
        pos[:, 2] = np.clip(pos[:, 2], 1000, 1e7)

        # emcee サンプラーの実行
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                                        args=(x_fit, y_fit, model, sigma))
        sampler.run_mcmc(pos, nsteps, progress=False)

        # バーンイン除去 & フラット化
        samples = sampler.get_chain(discard=burn_in, flat=True)
        # dV は index 2
        dV_samples = samples[:, 2]

        # パーセンタイル (16%, 50%, 84%)
        dV_16, dV_50, dV_84 = np.percentile(dV_samples, [16, 50, 84])
        dV_sigma = 0.5 * (dV_84 - dV_16)  # 1σ 相当

        R_val = ch_to_r.get(ch + 1, float(ch + 1))

        results.append({
            'ch': ch + 1,
            'R': R_val,
            'dV_median': dV_50,
            'dV_16': dV_16,
            'dV_84': dV_84,
            'dV_sigma': dV_sigma,
            'dV_samples': dV_samples,
            'all_samples': samples,
            'popt_curvefit': popt,
            'error': False
        })

        print(f"{ch+1:>3} | {R_val:>8.4f} | {dV_50:>12.1f} | {dV_16:>12.1f} | {dV_84:>12.1f} | {dV_sigma:>10.1f}")

        # ヒストグラム
        axes_hist[ch].hist(dV_samples, bins=50, color='steelblue', alpha=0.7, density=True)
        axes_hist[ch].axvline(dV_50, color='red', linewidth=2, label=f'median={dV_50/1000:.1f} km/s')
        axes_hist[ch].axvline(dV_16, color='orange', linestyle='--', linewidth=1, label=f'16%={dV_16/1000:.1f}')
        axes_hist[ch].axvline(dV_84, color='orange', linestyle='--', linewidth=1, label=f'84%={dV_84/1000:.1f}')
        axes_hist[ch].set_title(f'Ch {ch+1} (R={R_val:.2f} m)', fontsize=10)
        axes_hist[ch].set_xlabel('dV (m/s)', fontsize=8)
        axes_hist[ch].legend(fontsize=6, loc='upper right')
        axes_hist[ch].grid(True, linestyle=':', alpha=0.5)

    # y軸の高さを全サブプロットで統一
    y_max = max(ax.get_ylim()[1] for ax in axes_hist[:num_channels])
    for ch_i in range(num_channels):
        axes_hist[ch_i].set_ylim(0, y_max)

    fig_hist.suptitle('emcee: Posterior Distribution of dV per Channel', fontsize=14)
    fig_hist.tight_layout(rect=[0, 0, 1, 0.96])

    # --- 図1.5: 全チャンネルの dV 事後分布を1つのグラフに重ね合わせ ---
    fig_all, ax_all = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, num_channels))

    valid_for_overlay = [r for r in results if not r['error']]
    for i, res in enumerate(valid_for_overlay):
        ax_all.hist(res['dV_samples'], bins=50, color=colors[i], alpha=0.4, density=True,
                    label=f"Ch{res['ch']} (R={res['R']:.2f}m)")

    ax_all.set_xlabel('dV (m/s)', fontsize=12)
    ax_all.set_ylabel('Probability Density', fontsize=12)
    ax_all.set_title('emcee: dV Posterior Distributions (All Channels)', fontsize=14)
    ax_all.legend(fontsize=8, loc='upper right')
    ax_all.grid(True, linestyle=':', alpha=0.5)
    fig_all.tight_layout()

    # --- 図2: R vs dV (エラーバー付き) 比較プロット ---
    fig2, ax = plt.subplots(figsize=(10, 6))

    valid = [r for r in results if not r['error']]
    R_arr = [r['R'] for r in valid]
    dV_med = [r['dV_median'] for r in valid]
    dV_err_low = [r['dV_median'] - r['dV_16'] for r in valid]
    dV_err_high = [r['dV_84'] - r['dV_median'] for r in valid]
    dV_curvefit = [r['popt_curvefit'][2] for r in valid]  # curve_fit の dV

    # emcee の結果 (非対称エラーバー)
    ax.errorbar(R_arr, dV_med, yerr=[dV_err_low, dV_err_high], fmt='o',
                color='steelblue', markersize=8, capsize=5, capthick=1.5,
                elinewidth=1.5, label='emcee (median ± 16-84%)')

    # curve_fit の結果 (比較用)
    ax.plot(R_arr, dV_curvefit, 's', color='orange', markersize=8,
            markeredgecolor='black', markeredgewidth=0.5, label='curve_fit (MAP)')

    ax.set_xlabel('Major Radius R (m)', fontsize=12)
    ax.set_ylabel('Thermal Width dV (m/s)', fontsize=12)
    ax.set_title('R - dV Profile: emcee vs curve_fit', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.5)
    fig2.tight_layout()

    # --- CSV 保存 ---
    import csv
    csv_filename = 'dv_r_profile_emcee.csv'
    with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Channel', 'R (m)', 'dV_median (m/s)', 'dV_16 (m/s)', 'dV_84 (m/s)', 'dV_sigma (m/s)', 'dV_curvefit (m/s)'])
        for r in valid:
            writer.writerow([
                r['ch'],
                f"{r['R']:.4f}",
                f"{r['dV_median']:.4f}",
                f"{r['dV_16']:.4f}",
                f"{r['dV_84']:.4f}",
                f"{r['dV_sigma']:.4f}",
                f"{r['popt_curvefit'][2]:.4f}"
            ])
    # --- サンプルの保存 (後で確率密度プロットに使うため) ---
    npz_data = {}
    for r in valid:
        npz_data[f"ch_{r['ch']}_R"] = r['R']
        npz_data[f"ch_{r['ch']}_dV_samples"] = r['dV_samples']
    np.savez('emcee_samples.npz', **npz_data)
    print(f"[INFO] emcee のフルサンプルを 'emcee_samples.npz' として保存しました。")

    print(f"\n[INFO] emcee 結果を '{csv_filename}' として保存しました。")

    plt.show()


if __name__ == '__main__':
    solve_with_emcee()
