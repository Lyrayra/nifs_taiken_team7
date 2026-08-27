import numpy as np
import matplotlib.pyplot as plt
from sokudobunpu import generate_wavelength_distribution, velocity_to_lambda

def plot_assumed_wavelength_dist():
    # forward_model.py で仮定されたパラメータ
    lambda0 = 529.81891
    v0 = 0.0
    dV = 100000.0
    A = 3000.0
    
    # 速度軸の生成 (-300 km/s から +300 km/s まで)
    v = np.linspace(-300000, 300000, 1000)
    
    # 速度を波長(nm)に変換
    lam = velocity_to_lambda(v, lambda0=lambda0)
    
    # 波長分布関数 E(λ) の計算
    e_lam = generate_wavelength_distribution(lam, A, lambda0, v0, dV)
    
    plt.figure(figsize=(8, 5))
    
    # 横軸を波長にしてプロット
    plt.plot(lam, e_lam, '-', color='green', linewidth=2, 
             label=f'Assumed $E(\\lambda)$\n$\\lambda_0$ = {lambda0} nm\n$v_0$ = {v0} m/s\n$dV$ = {dV} m/s')
    
    # 領域の塗りつぶし
    plt.fill_between(lam, 0, e_lam, color='green', alpha=0.1)
    
    plt.xlabel('Wavelength $\\lambda$ (nm)', fontsize=12)
    plt.ylabel('Distribution $E(\\lambda)$', fontsize=12)
    plt.title('Assumed Wavelength Distribution (for Forward Model)', fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(fontsize=11)
    
    # 表示範囲を少し余裕を持たせる
    plt.xlim(lam.min(), lam.max())
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    plot_assumed_wavelength_dist()
