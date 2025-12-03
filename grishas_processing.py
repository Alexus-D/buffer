import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.signal
import main  # Import functions from main.py
import useful_funcs  # Import useful_funcs from main.py context (it's in the same folder)
from datetime import datetime

# Constants
PLATO_DB = -2.5
RES_MAGNITUDE_DB = -23.6  # Estimated from image
CAVITY_FREQ_GUESS = 3.70345  # From file at 2850 Oe
CAVITY_WIDTH_GUESS = 0.14919  # From file at 2850 Oe
RESULTS_DIR = os.path.join("grishas_results", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

def load_data(filepath):
    """Load frequency data from text file."""
    data = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
        # Skip header
        for line in lines[1:]:
            parts = line.strip().replace(',', '.').split()
            if len(parts) >= 5:
                try:
                    field = float(parts[0])
                    f1 = float(parts[1])
                    f2 = float(parts[2])
                    bw1 = float(parts[3])
                    bw2 = float(parts[4])
                    
                    # Check for NaNs
                    if np.isnan(field) or np.isnan(f1) or np.isnan(f2) or np.isnan(bw1) or np.isnan(bw2):
                        continue
                        
                    data.append([field, f1, f2, bw1, bw2])
                except ValueError:
                    continue
    return np.array(data)

def process_grishas_data():
    # 1. Load Data
    data_array = load_data(os.path.join("grishas_data", "Frequences.txt"))
    fields = data_array[:, 0]
    freq1 = data_array[:, 1]
    freq2 = data_array[:, 2]
    bw1 = data_array[:, 3]
    bw2 = data_array[:, 4]

    # Outlier removal
    def get_outlier_mask(data, kernel_size=5, threshold=5):
        # Pad to handle edges correctly with medfilt (which zero-pads)
        pad_size = kernel_size // 2
        padded = np.pad(data, (pad_size, pad_size), mode='edge')
        smooth = scipy.signal.medfilt(padded, kernel_size)[pad_size:-pad_size]
        
        residuals = np.abs(data - smooth)
        sigma = np.median(residuals) * 1.4826  # Robust sigma (MAD)
        if sigma == 0:
            sigma = np.std(residuals)
        if sigma == 0:
            return np.zeros(len(data), dtype=bool)
            
        return residuals > (threshold * sigma)

    # Detect outliers
    mask_f1 = get_outlier_mask(freq1)
    mask_f2 = get_outlier_mask(freq2)
    mask_bw1 = get_outlier_mask(bw1)
    mask_bw2 = get_outlier_mask(bw2)
    
    # Combine masks
    bad_indices = mask_f1 | mask_f2 | mask_bw1 | mask_bw2
    
    if np.any(bad_indices):
        print(f"Removing {np.sum(bad_indices)} outlier points.")
        fields = fields[~bad_indices]
        freq1 = freq1[~bad_indices]
        freq2 = freq2[~bad_indices]
        bw1 = bw1[~bad_indices]
        bw2 = bw2[~bad_indices]

    # Plot filtered data
    plt.figure(figsize=(10, 6))
    plt.plot(fields, freq1, 'b-', label='Freq 1')
    plt.plot(fields, freq2, 'r-', label='Freq 2')
    plt.xlabel('Field (Oe)')
    plt.ylabel('Frequency (GHz)')
    plt.title('Filtered Data')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(fields, bw1, 'b-', label='BW 1')
    plt.plot(fields, bw2, 'r-', label='BW 2')
    plt.xlabel('Field (Oe)')
    plt.ylabel('Bandwidth (GHz)')
    plt.title('Filtered Bandwidth Data')
    plt.legend()
    plt.show()

    # 2. Construct own_modes
    # Mode = Freq - i * (BW / 2)
    mode1 = freq1 - 1j * (bw1 / 2)
    mode2 = freq2 - 1j * (bw2 / 2)
    
    own_modes = {
        'fields': [fields],  # extract_coupling_params expects list of arrays? No, it does fields = own_modes['fields'][0]
        'modes': [mode1, mode2]
    }

    # 3. Construct resonator params
    # Convert dB to linear
    plato_linear = 10**(PLATO_DB / 20)
    res_magnitude_linear = 10**(RES_MAGNITUDE_DB / 20)
    
    # Use estimate_cavity_params from useful_funcs
    # Note: useful_funcs is imported in main, but we can also import it directly.
    # main.useful_funcs points to the module.
    
    est_params = main.useful_funcs.estimate_cavity_params(
        res_magnitude_linear,
        CAVITY_FREQ_GUESS,
        CAVITY_WIDTH_GUESS,
        plato_linear
    )
    
    resonator = {
        'resonator_params': est_params
    }
    
    # 3.5 Set Magnon Calibration
    # User request: Two points calibration
    # (2855, 3.39) and (3049, 3.879)
    import pickle
    
    # Use a separate buffer file for Grisha's processing to avoid conflicts/corruption
    GRISHAS_BUFFER_FILE = "grishas_buffer.pkl"
    main.BUFFER_DATA_FILE = GRISHAS_BUFFER_FILE
 
    # Calibration points
    h1, f1 = 2854.0, 3.387
    h2, f2 = 3050.0, 3.883
    
    gamma_g = (f2 - f1) / (h2 - h1)
    offset = f1 - gamma_g * h1
    
    # Update global config so main.py uses the correct gamma
    main.config_physics.GYROMAGNETIC_RATIO = gamma_g
    
    calibration_params = {
        'offset': offset,
        'magnon_freqs_calibrated': None # Will be recalculated
    }
    
    # Save to buffer file so extract_coupling_params picks it up
    buffer_file = main.BUFFER_DATA_FILE
    try:
        with open(buffer_file, 'rb') as f:
            buffer_data = pickle.load(f)
    except (FileNotFoundError, EOFError, ImportError, ModuleNotFoundError):
        # ImportError/ModuleNotFoundError can happen if pickle was saved with different numpy version
        buffer_data = {}
        
    buffer_data['magnon_calibration'] = calibration_params
    with open(buffer_file, 'wb') as f:
        pickle.dump(buffer_data, f)
        
    print(f"Set calibration: Gamma = {gamma_g:.6f} GHz/Oe, Offset = {offset:.6f} GHz")
    print(f"Using buffer file: {buffer_file}")
    
    # 4. Extract Coupling Params
    print("Extracting coupling parameters...")
    coupling_params = main.extract_coupling_params(resonator, own_modes, grisha=True, magnon_calibration={'offset': offset, 'gamma_g': gamma_g})
    
    # 5. Get General Parameters
    print("Calculating general parameters...")
    general_params = main.get_general_parameters(coupling_params)
    
    # Calculate derived parameters
    general_params['cavity_width'] = 2 * (general_params['kappa'] + general_params['beta'])
    general_params['Q_L'] = general_params['cavity_freq'] / general_params['cavity_width']
    # Cooperativity C = J^2 / (kappa_tot * gamma_tot) ? 
    # Assuming kappa and gamma are HWHM.
    # Let's use J^2 / (kappa * gamma) as a simple metric or J^2 / ((kappa+beta)*(gamma+alpha))
    # I'll use the total linewidths product.
    kappa_tot = general_params['kappa'] + general_params['beta']
    gamma_tot = general_params['gamma_mean'] + general_params['alpha_mean']
    general_params['cooperativity'] = general_params['J_mean']**2 / (kappa_tot * gamma_tot)

    # 6. Reconstruct Own Modes
    print("Reconstructing eigenmodes...")
    reconstructed_modes = main.reconstruct_own_modes(general_params, coupling_params)
    
    # 7. Save Results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Save General Parameters
    with open(os.path.join(RESULTS_DIR, 'general_parameters.txt'), 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("УСРЕДНЕННЫЕ ПАРАМЕТРЫ СИСТЕМЫ\n")
        f.write("="*70 + "\n\n")
        f.write("ПАРАМЕТРЫ РЕЗОНАТОРА:\n")
        f.write("-"*70 + "\n")
        f.write(f"Частота:                ω_c = {general_params['cavity_freq']:.6f} ГГц\n")
        f.write(f"Внешнее затухание:      κ   = {general_params['kappa']:.6f} ГГц\n")
        f.write(f"Внутреннее затухание:   β   = {general_params['beta']:.6f} ГГц\n")
        f.write(f"Полная ширина линии:    Δω  = {general_params['cavity_width']:.6f} ГГц\n")
        f.write(f"Нагруженная добротность: Q_L = {general_params['Q_L']:.2f}\n\n")
        f.write("ПАРАМЕТРЫ СВЯЗИ:\n")
        f.write("-"*70 + "\n")
        f.write(f"Когерентная связь:      J   = {general_params['J_mean']:.6f} ± {general_params['J_err']:.6f} ГГц\n")
        f.write(f"Диссипативная связь:    Γ   = {general_params['Gamma_mean']:.6f} ± {general_params['Gamma_err']:.6f} ГГц\n")
        f.write(f"Купертивность (C):      C   = {general_params['cooperativity']:.4f}\n\n")
        f.write("ПАРАМЕТРЫ МАГНОНОВ:\n")
        f.write("-"*70 + "\n")
        f.write(f"Средняя частота:        ω_m = {general_params['magnon_freq_mean']:.6f} ± {general_params['magnon_freq_err']:.6f} ГГц\n")
        f.write(f"Собственное затухание:  α   = {general_params['alpha_mean']:.6f} ± {general_params['alpha_err']:.6f} ГГц\n")
        f.write(f"Индуцированное затух.:  γ   = {general_params['gamma_mean']:.6f} ± {general_params['gamma_err']:.6f} ГГц\n")
        f.write(f"Калибровочное смещение: Δω  = {coupling_params['magnon_offset']:.6f} ГГц\n\n")
        f.write("="*70 + "\n")

    # Save Field Dependent Parameters
    with open(os.path.join(RESULTS_DIR, 'field_dependent_parameters.txt'), 'w', encoding='utf-8') as f:
        f.write("="*120 + "\n")
        f.write("ПАРАМЕТРЫ СВЯЗИ И МАГНОНОВ ДЛЯ КАЖДОГО ЗНАЧЕНИЯ МАГНИТНОГО ПОЛЯ\n")
        f.write("="*120 + "\n\n")
        header = f"{'Поле (Э)':>12} {'ω_m (ГГц)':>12} {'ω_m_эксп':>12} {'J (ГГц)':>12} {'Γ (ГГц)':>12} {'α (ГГц)':>12} {'γ (ГГц)':>12}\n"
        f.write(header)
        f.write("-"*120 + "\n")
        for i, field in enumerate(fields):
            line = f"{field:12.2f} {coupling_params['magnon_freq'][i]:12.6f} {coupling_params['magnon_freq_experimental'][i]:12.6f} "
            line += f"{coupling_params['J'][i]:12.6f} {coupling_params['Gamma'][i]:12.6f} "
            line += f"{coupling_params['alpha'][i]:12.6f} {coupling_params['gamma'][i]:12.6f}\n"
            f.write(line)

    # Save Eigenmodes Data
    exp_mode_1 = own_modes['modes'][0]
    exp_mode_2 = own_modes['modes'][1]
    # Swap + and - modes as per user request
    rec_mode_plus = np.array(reconstructed_modes['mode_minus'])
    rec_mode_minus = np.array(reconstructed_modes['mode_plus'])
    
    with open(os.path.join(RESULTS_DIR, 'eigenmodes_data.txt'), 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write("ДАННЫЕ СОБСТВЕННЫХ МОД\n")
        f.write("="*100 + "\n\n")
        header = f"{'Поле (Э)':>12} {'Mode1_Re':>12} {'Mode1_Im':>12} {'Mode2_Re':>12} {'Mode2_Im':>12} "
        header += f"{'Rec+_Re':>12} {'Rec+_Im':>12} {'Rec-_Re':>12} {'Rec-_Im':>12}\n"
        f.write(header)
        f.write("-"*100 + "\n")
        for i, field in enumerate(fields):
            line = f"{field:12.2f} {exp_mode_1.real[i]:12.6f} {exp_mode_1.imag[i]:12.6f} "
            line += f"{exp_mode_2.real[i]:12.6f} {exp_mode_2.imag[i]:12.6f} "
            line += f"{rec_mode_plus.real[i]:12.6f} {rec_mode_plus.imag[i]:12.6f} "
            line += f"{rec_mode_minus.real[i]:12.6f} {rec_mode_minus.imag[i]:12.6f}\n"
            f.write(line)

    # 8. Plot System Parameters
    plot_system_parameters(fields, coupling_params, general_params, own_modes)
    
    # 9. Plot Eigenmodes Comparison
    plot_eigenmodes_comparison(fields, own_modes, reconstructed_modes)
    
    # 10. Plot Anticrossing (Reconstructed)
    plot_anticrossing(general_params, coupling_params)

def plot_system_parameters(fields, coupling_params, general_params, own_modes):
    fig = plt.figure(figsize=(14, 10))
    plt.rcParams.update({'font.size': 10})
    
    mode_1 = own_modes['modes'][0]
    mode_2 = own_modes['modes'][1]
    
    # 1. Resonator Params
    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(fields, coupling_params['kappa'], 'b-', linewidth=2, label='κ (external)')
    ax1.plot(fields, coupling_params['beta'], 'r-', linewidth=2, label='β (internal)')
    ax1.axhline(y=general_params['kappa'], color='b', linestyle='--', alpha=0.7, label=f'κ = {general_params["kappa"]:.4f}')
    ax1.axhline(y=general_params['beta'], color='r', linestyle='--', alpha=0.7, label=f'β = {general_params["beta"]:.4f}')
    ax1.set_title('Resonator Loss Rates', fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlabel('Magnetic Field (Oe)', fontsize=10)
    ax1.set_ylabel('Loss Rate (GHz)', fontsize=10)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 2. Coupling Params
    ax2 = plt.subplot(3, 2, 2)
    ax2.plot(fields, coupling_params['J'], 'g-', linewidth=2, label='J (coherent)')
    ax2.plot(fields, coupling_params['Gamma'], 'm-', linewidth=2, label='Γ (dissipative)')
    ax2.axhline(y=general_params['J_mean'], color='g', linestyle='--', alpha=0.7, label=f'Mean J = {general_params["J_mean"]:.4f}')
    ax2.axhline(y=general_params['Gamma_mean'], color='m', linestyle='--', alpha=0.7, label=f'Mean Γ = {general_params["Gamma_mean"]:.4f}')
    ax2.set_title('Coupling Parameters', fontsize=12, fontweight='bold', pad=10)
    ax2.set_xlabel('Magnetic Field (Oe)', fontsize=10)
    ax2.set_ylabel('Coupling Strength (GHz)', fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 3. Magnon Params
    ax3 = plt.subplot(3, 2, 3)
    ax3.plot(fields, coupling_params['alpha'], 'b-', linewidth=2, label='α (intrinsic)')
    ax3.plot(fields, coupling_params['gamma'], 'r-', linewidth=2, label='γ (induced)')
    ax3.axhline(y=general_params['alpha_mean'], color='b', linestyle='--', alpha=0.7, label=f'Mean α = {general_params["alpha_mean"]:.4f}')
    ax3.axhline(y=general_params['gamma_mean'], color='r', linestyle='--', alpha=0.7, label=f'Mean γ = {general_params["gamma_mean"]:.4f}')
    ax3.set_title('Magnon Loss Rates', fontsize=12, fontweight='bold', pad=10)
    ax3.set_xlabel('Magnetic Field (Oe)', fontsize=10)
    ax3.set_ylabel('Loss Rate (GHz)', fontsize=10)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # 4. Frequencies
    ax4 = plt.subplot(3, 2, 4)
    ax4.plot(fields, coupling_params['magnon_freq'], 'b-', linewidth=2, label='Magnon (calibrated)')
    ax4.plot(fields, coupling_params['magnon_freq_experimental'], 'b:', linewidth=1.5, alpha=0.7, label='Magnon (experimental)')
    ax4.axhline(y=general_params['cavity_freq'], color='r', linestyle='--', linewidth=2, label='Cavity')
    ax4.set_title('Bare Frequencies', fontsize=12, fontweight='bold', pad=10)
    ax4.set_xlabel('Magnetic Field (Oe)', fontsize=10)
    ax4.set_ylabel('Frequency (GHz)', fontsize=10)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # 5. Eigenmode Frequencies
    ax5 = plt.subplot(3, 2, 5)
    ax5.plot(fields, mode_1.real, 'b-', linewidth=2, label='Mode 1')
    ax5.plot(fields, mode_2.real, 'r-', linewidth=2, label='Mode 2')
    ax5.set_title('Eigenmode Frequencies', fontsize=12, fontweight='bold', pad=10)
    ax5.set_xlabel('Magnetic Field (Oe)', fontsize=10)
    ax5.set_ylabel('Frequency (GHz)', fontsize=10)
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    
    # 6. Eigenmode Damping
    ax6 = plt.subplot(3, 2, 6)
    ax6.plot(fields, mode_1.imag, 'b-', linewidth=2, label='Mode 1')
    ax6.plot(fields, mode_2.imag, 'r-', linewidth=2, label='Mode 2')
    ax6.set_title('Eigenmode Damping Rates', fontsize=12, fontweight='bold', pad=10)
    ax6.set_xlabel('Magnetic Field (Oe)', fontsize=10)
    ax6.set_ylabel('Damping Rate (GHz)', fontsize=10)
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout(pad=2.0)
    plt.savefig(os.path.join(RESULTS_DIR, 'system_parameters.png'), dpi=300, bbox_inches='tight')
    plt.show()

def plot_eigenmodes_comparison(fields, own_modes, reconstructed_modes):
    fig = plt.figure(figsize=(14, 10))
    
    exp_mode_1 = own_modes['modes'][0]
    exp_mode_2 = own_modes['modes'][1]
    # Swap + and - modes as per user request
    rec_mode_plus = np.array(reconstructed_modes['mode_minus'])
    rec_mode_minus = np.array(reconstructed_modes['mode_plus'])
    
    # 1. Frequencies
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(fields, exp_mode_1.real, 'b-', linewidth=2.5, label='Exp. Mode 1', alpha=0.8)
    ax1.plot(fields, exp_mode_2.real, 'r-', linewidth=2.5, label='Exp. Mode 2', alpha=0.8)
    ax1.plot(fields, rec_mode_plus.real, 'b--', linewidth=1.5, label='Rec. Mode +')
    ax1.plot(fields, rec_mode_minus.real, 'r--', linewidth=1.5, label='Rec. Mode −')
    ax1.set_xlabel('Magnetic Field (Oe)', fontsize=11)
    ax1.set_ylabel('Frequency (GHz)', fontsize=11)
    ax1.set_title('Eigenmode Frequencies', fontsize=12, fontweight='bold', pad=10)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 2. Damping
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(fields, exp_mode_1.imag, 'b-', linewidth=2.5, label='Exp. Mode 1', alpha=0.8)
    ax2.plot(fields, exp_mode_2.imag, 'r-', linewidth=2.5, label='Exp. Mode 2', alpha=0.8)
    ax2.plot(fields, rec_mode_plus.imag, 'b--', linewidth=1.5, label='Rec. Mode +')
    ax2.plot(fields, rec_mode_minus.imag, 'r--', linewidth=1.5, label='Rec. Mode −')
    ax2.set_xlabel('Magnetic Field (Oe)', fontsize=11)
    ax2.set_ylabel('Damping Rate (GHz)', fontsize=11)
    ax2.set_title('Eigenmode Damping Rates', fontsize=12, fontweight='bold', pad=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 3. Frequency Residuals
    ax3 = plt.subplot(2, 2, 3)
    diff_freq_1 = (exp_mode_1.real - rec_mode_plus.real) * 1000
    diff_freq_2 = (exp_mode_2.real - rec_mode_minus.real) * 1000
    ax3.plot(fields, diff_freq_1, 'b-', linewidth=2, label=f'Mode 1 − Mode + (RMS={np.sqrt(np.mean(diff_freq_1**2)):.2f} MHz)')
    ax3.plot(fields, diff_freq_2, 'r-', linewidth=2, label=f'Mode 2 − Mode − (RMS={np.sqrt(np.mean(diff_freq_2**2)):.2f} MHz)')
    ax3.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax3.set_xlabel('Magnetic Field (Oe)', fontsize=11)
    ax3.set_ylabel('Frequency Difference (MHz)', fontsize=11)
    ax3.set_title('Frequency Residuals', fontsize=12, fontweight='bold', pad=10)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # 4. Damping Residuals
    ax4 = plt.subplot(2, 2, 4)
    diff_damp_1 = (exp_mode_1.imag - rec_mode_plus.imag) * 1000
    diff_damp_2 = (exp_mode_2.imag - rec_mode_minus.imag) * 1000
    ax4.plot(fields, diff_damp_1, 'b-', linewidth=2, label=f'Mode 1 − Mode + (RMS={np.sqrt(np.mean(diff_damp_1**2)):.2f} MHz)')
    ax4.plot(fields, diff_damp_2, 'r-', linewidth=2, label=f'Mode 2 − Mode − (RMS={np.sqrt(np.mean(diff_damp_2**2)):.2f} MHz)')
    ax4.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax4.set_xlabel('Magnetic Field (Oe)', fontsize=11)
    ax4.set_ylabel('Damping Difference (MHz)', fontsize=11)
    ax4.set_title('Damping Residuals', fontsize=12, fontweight='bold', pad=10)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout(pad=2.0)
    plt.savefig(os.path.join(RESULTS_DIR, 'modes_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()

def plot_anticrossing(general_params, coupling_params):
    # Define grid
    fields = np.linspace(2850, 3050, 201)
    freqs = np.linspace(3.0, 4.0, 201)
    
    # Extract params
    alpha = general_params['alpha_mean']
    beta = general_params['beta']
    kappa = general_params['kappa']
    gamma = general_params['gamma_mean']
    J = general_params['J_mean']
    G = general_params['Gamma_mean']
    cavity_freq = general_params['cavity_freq']
    plato = general_params['plato']
    
    # Magnon calibration
    offset = coupling_params['magnon_offset']
    gamma_g = main.config_physics.GYROMAGNETIC_RATIO
    
    s_param_reconstructed = np.zeros((len(fields), len(freqs)), dtype=complex)
    
    for i, field in enumerate(fields):
        magnon_freq = gamma_g * field + offset
        
        cavity_delta = freqs - cavity_freq
        magnon_delta = freqs - magnon_freq
        coupling = J - 1j * G
        
        cavity_term = 1j * cavity_delta - (kappa + beta)
        magnon_term = 1j * magnon_delta - (gamma + alpha)
        
        response = plato + (kappa / (cavity_term - (coupling**2) / magnon_term))
        s_param_reconstructed[i, :] = response
        
    # Plot
    fig = plt.figure(figsize=(8, 6))
    s_rec_db = 20 * np.log10(np.abs(s_param_reconstructed))
    
    plt.contourf(fields, freqs, s_rec_db.T, levels=100, cmap='jet')
    plt.colorbar(label='|S21| (dB)')
    plt.xlabel('Magnetic Field (Oe)')
    plt.ylabel('Frequency (GHz)')
    plt.title('Reconstructed Anticrossing')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'anticrossing.png'), dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    process_grishas_data()
