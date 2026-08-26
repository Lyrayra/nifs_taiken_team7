import re

with open('20260826/forward_model.py', 'r') as f:
    content = f.read()

new_func = """def load_dat_spectrum(file_path):
    import numpy as np
    data_list = []
    with open(file_path, 'r') as f:
        is_data_section = False
        for line in f:
            line = line.strip()
            if line == '[Data]' or line == '# [Data]':
                is_data_section = True
                continue
            
            if is_data_section and line and not line.startswith('#'):
                parts = [float(v) for v in line.split(',')]
                data_list.append(parts)
                
    raw_data = np.array(data_list)
    
    # 0 column is xpix, 1-12 columns are stripes
    spectrum_data = np.mean(raw_data[:, 1:], axis=1) # Average over the 12 stripes
    
    # Wavelength axis
    x_max = len(spectrum_data)
    pixel_peaks = np.array([12, 33, 92, 111])
    wavelength_peaks = np.array([530.47580, 529.81891, 528.03, 527.40393])
    poly = np.poly1d(np.polyfit(pixel_peaks, wavelength_peaks, 1))
    wavelength_axis = np.linspace(poly(0), poly(x_max - 1), x_max)
    
    return wavelength_axis, spectrum_data

def plot_forward_model():"""

content = content.replace("def plot_forward_model():", new_func)

plot_code_old = """    # プロット
    plt.figure(figsize=(10, 5))
    plt.plot(wl, m_spec, label='Measured $M(\lambda)$', color='blue', alpha=0.5)
    plt.plot(wl, e_spec, label='Theory $E(\lambda)$ (No Inst. Broadening)', color='green', linestyle='--')
    plt.plot(wl, d_spec, label='Forward Model $D(\lambda) = E*I + bg$', color='red', linewidth=2)"""

plot_code_new = """    # 2つ目のデータ (.dat) を読み込む
    dat_file_path = 'lhdcxs9a_img_sig@189129_t4.44s.dat'
    dat_wl, dat_spec = load_dat_spectrum(dat_file_path)

    # プロット
    plt.figure(figsize=(10, 5))
    plt.plot(wl, m_spec, label=r'Measured $M(\lambda)$', color='blue', alpha=0.5)
    plt.plot(wl, e_spec, label=r'Theory $E(\lambda)$ (No Inst. Broadening)', color='green', linestyle='--')
    plt.plot(wl, d_spec, label=r'Forward Model $D(\lambda) = E*I + bg$', color='red', linewidth=2)
    plt.plot(dat_wl, dat_spec, label=r'New Data (t=4.44s)', color='purple', linewidth=1.5, linestyle='-.')"""

content = content.replace(plot_code_old, plot_code_new)

# Fix invalid escape sequences while we are at it
content = content.replace("'Measured $M(\lambda)$'", "r'Measured $M(\lambda)$'")
content = content.replace("'Theory $E(\lambda)$ (No Inst. Broadening)'", "r'Theory $E(\lambda)$ (No Inst. Broadening)'")
content = content.replace("'Forward Model $D(\lambda) = E*I + bg$'", "r'Forward Model $D(\lambda) = E*I + bg$'")


with open('20260826/forward_model.py', 'w') as f:
    f.write(content)
