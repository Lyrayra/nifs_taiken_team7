import re

with open('20260826/forward_model.py', 'r') as f:
    content = f.read()

# Define the old plot block using regex to capture everything from "# プロット" to "plt.show()"
old_plot_pattern = re.compile(r'    # プロット\n.*plt\.show\(\)', re.DOTALL)

new_plot_block = """    # プロット
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    ax1.plot(wl, m_spec, label=r'Measured $M(\lambda)$', color='blue', alpha=0.5)
    ax1.plot(wl, e_spec, label=r'Theory $E(\lambda)$ (No Inst. Broadening)', color='green', linestyle='--')
    ax1.plot(wl, d_spec, label=r'Forward Model $D(\lambda) = E*I + bg$', color='red', linewidth=2)
    
    # スケールが違うため、第2のY軸を作成してプロット
    ax2 = ax1.twinx()
    ax2.plot(dat_wl, dat_spec, label=r'New Data (t=4.44s)', color='purple', linewidth=1.5, linestyle='-.')
    
    # 解析対象のピーク周辺を拡大表示
    ax1.set_xlim(target_line - 0.5, target_line + 0.5)
    ax1.set_ylim(bg - 50, np.nanmax(m_spec[(wl > target_line - 0.5) & (wl < target_line + 0.5)]) * 1.1)
    
    # 新しいデータのY軸の範囲も少し余裕を持たせる（0〜最大値の1.1倍）
    dat_max = np.nanmax(dat_spec[(dat_wl > target_line - 0.5) & (dat_wl < target_line + 0.5)])
    if dat_max > 0:
        ax2.set_ylim(0, dat_max * 1.2)

    ax1.set_xlabel('Wavelength (nm)')
    ax1.set_ylabel('Count (a260825_img)')
    ax2.set_ylabel('Signal (New Data)')
    plt.title('Forward Problem: Model vs Measurement')
    
    # 凡例を統合する
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    ax1.grid(True, linestyle=':')
    plt.tight_layout()
    plt.show()"""

# Replace
if old_plot_pattern.search(content):
    content = old_plot_pattern.sub(new_plot_block, content)
    with open('20260826/forward_model.py', 'w') as f:
        f.write(content)
    print("Success")
else:
    print("Pattern not found")

