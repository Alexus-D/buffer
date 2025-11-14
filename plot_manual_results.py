"""
Временный скрипт для построения графиков из существующих результатов ручного выбора пиков

Использование:
    python plot_manual_results.py results/manual_peak_selection_YYYYMMDD_HHMMSS
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import data_io
import parameter_extraction


def plot_all_parameters(results_dir, peak_df, coupling_df, cavity_params):
    """Построить все графики параметров"""
    print("\nGenerating comprehensive parameter plots...")
    
    # Создать большой набор графиков (3x3)
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle('Manual Peak Selection: All Parameters vs Field', fontsize=16, fontweight='bold')
    
    fields = peak_df['Field(Э)'].values
    
    # График 1: Частоты пиков (резонансные частоты)
    axes[0, 0].plot(fields, peak_df['Freq1(ГГц)'], 'ro-', label='Peak 1 (ω₊)', markersize=6, linewidth=2)
    axes[0, 0].plot(fields, peak_df['Freq2(ГГц)'], 'bo-', label='Peak 2 (ω₋)', markersize=6, linewidth=2)
    axes[0, 0].set_xlabel('Field (Oe)', fontsize=10)
    axes[0, 0].set_ylabel('Frequency (GHz)', fontsize=10)
    axes[0, 0].set_title('Resonance Frequencies (Peaks)', fontsize=11, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # График 2: Ширины пиков
    axes[0, 1].plot(fields, peak_df['Width1(ГГц)'] * 1000, 'ro-', label='Peak 1', markersize=6, linewidth=2)
    axes[0, 1].plot(fields, peak_df['Width2(ГГц)'] * 1000, 'bo-', label='Peak 2', markersize=6, linewidth=2)
    axes[0, 1].set_xlabel('Field (Oe)', fontsize=10)
    axes[0, 1].set_ylabel('Width (MHz)', fontsize=10)
    axes[0, 1].set_title('Peak Widths', fontsize=11, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # График 3: Константа связи J
    axes[0, 2].plot(coupling_df['field'], coupling_df['J'] * 1000, 'go-', markersize=6, linewidth=2)
    axes[0, 2].set_xlabel('Field (Oe)', fontsize=10)
    axes[0, 2].set_ylabel('J (MHz)', fontsize=10)
    axes[0, 2].set_title('Coupling Constant J', fontsize=11, fontweight='bold')
    axes[0, 2].grid(True, alpha=0.3)
    
    # График 4: Gamma (комбинированные потери)
    axes[1, 0].plot(coupling_df['field'], coupling_df['Gamma'] * 1000, 'mo-', markersize=6, linewidth=2)
    axes[1, 0].set_xlabel('Field (Oe)', fontsize=10)
    axes[1, 0].set_ylabel('Γ (MHz)', fontsize=10)
    axes[1, 0].set_title('Combined Damping Γ', fontsize=11, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # График 5: gamma (магнонные потери)
    axes[1, 1].plot(coupling_df['field'], coupling_df['gamma'] * 1000, 'co-', markersize=6, linewidth=2)
    axes[1, 1].set_xlabel('Field (Oe)', fontsize=10)
    axes[1, 1].set_ylabel('γ (MHz)', fontsize=10)
    axes[1, 1].set_title('Magnon Damping γ', fontsize=11, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    # График 6: alpha (гильбертово затухание)
    axes[1, 2].plot(coupling_df['field'], coupling_df['alpha'] * 1000, 'orange', marker='o', markersize=6, linewidth=2)
    axes[1, 2].set_xlabel('Field (Oe)', fontsize=10)
    axes[1, 2].set_ylabel('α (MHz)', fontsize=10)
    axes[1, 2].set_title('Gilbert Damping α', fontsize=11, fontweight='bold')
    axes[1, 2].grid(True, alpha=0.3)
    
    # График 7: omega_c (парциальная частота резонатора)
    if cavity_params is not None:
        wc = cavity_params['wc']
        axes[2, 0].axhline(y=wc, color='red', linestyle='--', linewidth=2, label=f'ωc = {wc:.4f} GHz')
        axes[2, 0].set_xlabel('Field (Oe)', fontsize=10)
        axes[2, 0].set_ylabel('ωc (GHz)', fontsize=10)
        axes[2, 0].set_title('Cavity Frequency ωc', fontsize=11, fontweight='bold')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        axes[2, 0].set_xlim(fields.min(), fields.max())
    
    # График 8: omega_m_real (парциальная частота магнонов)
    axes[2, 1].plot(coupling_df['field'], coupling_df['omega_m_real'], 'purple', marker='o', markersize=6, linewidth=2)
    axes[2, 1].set_xlabel('Field (Oe)', fontsize=10)
    axes[2, 1].set_ylabel('ωm (GHz)', fontsize=10)
    axes[2, 1].set_title('Magnon Frequency ωm (Real Part)', fontsize=11, fontweight='bold')
    axes[2, 1].grid(True, alpha=0.3)
    
    # График 9: kappa и beta
    if cavity_params is not None:
        kappa = cavity_params['kappa']
        beta = cavity_params['beta']
        axes[2, 2].axhline(y=kappa * 1000, color='blue', linestyle='--', linewidth=2, label=f'κ = {kappa*1000:.2f} MHz')
        axes[2, 2].axhline(y=beta * 1000, color='green', linestyle='--', linewidth=2, label=f'β = {beta*1000:.2f} MHz')
        axes[2, 2].set_xlabel('Field (Oe)', fontsize=10)
        axes[2, 2].set_ylabel('Damping (MHz)', fontsize=10)
        axes[2, 2].set_title('Cavity Damping κ & β', fontsize=11, fontweight='bold')
        axes[2, 2].legend()
        axes[2, 2].grid(True, alpha=0.3)
        axes[2, 2].set_xlim(fields.min(), fields.max())
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # Сохранить график
    plot_file = os.path.join(results_dir, "all_parameters.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"All parameters plot saved to: {plot_file}")
    
    plt.show()


def plot_model_comparison(results_dir, data_file, coupling_df, cavity_params):
    """Построить сравнение экспериментальных данных с моделью"""
    print("\nGenerating model comparison (contour plots)...")
    
    # Загрузить данные
    print(f"Loading data from: {data_file}")
    data = data_io.load_s_parameter_data(data_file)
    freq = data['freq']
    field = data['field']
    s_param = data['s_param']
    s_mag = np.abs(s_param)
    
    # Восстановить модель из параметров
    print("Computing model S-parameters...")
    s_model = parameter_extraction.compute_s_parameter_model(
        freq=freq,
        field=field,
        cavity_params=cavity_params,
        coupling_results=coupling_df.to_dict('records'),
        s_type='S21'
    )
    
    # Вычислить модуль
    s_model_mag = np.abs(s_model)
    
    # Вычислить ошибку
    error = s_model_mag - s_mag
    
    # Создать 3 контурных графика
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Manual Peak Selection: Model vs Experiment', fontsize=14, fontweight='bold')
    
    FREQ_GRID, FIELD_GRID = np.meshgrid(freq, field)
    
    # График 1: Экспериментальные данные
    im1 = axes[0].contourf(FREQ_GRID, FIELD_GRID, s_mag, levels=50, cmap='viridis')
    axes[0].set_xlabel('Frequency (GHz)', fontsize=11)
    axes[0].set_ylabel('Field (Oe)', fontsize=11)
    axes[0].set_title('Experimental |S|', fontsize=12, fontweight='bold')
    plt.colorbar(im1, ax=axes[0], label='|S|')
    
    # График 2: Модель
    im2 = axes[1].contourf(FREQ_GRID, FIELD_GRID, s_model_mag, levels=50, cmap='viridis')
    axes[1].set_xlabel('Frequency (GHz)', fontsize=11)
    axes[1].set_ylabel('Field (Oe)', fontsize=11)
    axes[1].set_title('Model |S|', fontsize=12, fontweight='bold')
    plt.colorbar(im2, ax=axes[1], label='|S|')
    
    # График 3: Ошибка
    error_max = max(abs(error.min()), abs(error.max()))
    im3 = axes[2].contourf(FREQ_GRID, FIELD_GRID, error, levels=50, cmap='RdBu_r', 
                           vmin=-error_max, vmax=error_max)
    axes[2].set_xlabel('Frequency (GHz)', fontsize=11)
    axes[2].set_ylabel('Field (Oe)', fontsize=11)
    rms_error = np.sqrt(np.mean(error**2))
    axes[2].set_title(f'Error (Model - Exp), RMS={rms_error:.4f}', fontsize=12, fontweight='bold')
    plt.colorbar(im3, ax=axes[2], label='Error')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # Сохранить график
    plot_file = os.path.join(results_dir, "model_comparison.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Model comparison plot saved to: {plot_file}")
    
    plt.show()


def main(results_dir):
    """
    Главная функция
    
    Parameters:
    -----------
    results_dir : str
        Путь к директории с результатами ручного выбора пиков
    """
    print("=" * 70)
    print("PLOTTING MANUAL PEAK SELECTION RESULTS")
    print("=" * 70)
    print(f"\nResults directory: {results_dir}")
    
    # Проверить существование директории
    if not os.path.exists(results_dir):
        print(f"\nERROR: Directory not found: {results_dir}")
        return
    
    # Загрузить CSV файлы
    peak_file = os.path.join(results_dir, "peak_parameters.csv")
    coupling_file = os.path.join(results_dir, "coupling_parameters.csv")
    
    if not os.path.exists(peak_file):
        print(f"\nERROR: Peak parameters file not found: {peak_file}")
        return
    
    if not os.path.exists(coupling_file):
        print(f"\nERROR: Coupling parameters file not found: {coupling_file}")
        return
    
    print(f"\nLoading peak parameters from: {peak_file}")
    peak_df = pd.read_csv(peak_file)
    
    print(f"Loading coupling parameters from: {coupling_file}")
    coupling_df = pd.read_csv(coupling_file)
    
    print(f"\nPeak parameters shape: {peak_df.shape}")
    print(f"Coupling parameters shape: {coupling_df.shape}")
    
    # Извлечь параметры резонатора из coupling_df (они должны быть одинаковыми)
    # Или запросить у пользователя
    print("\n" + "=" * 70)
    print("CAVITY PARAMETERS")
    print("=" * 70)
    print("\nEnter cavity parameters (or press Enter to skip):")
    
    wc_input = input("  wc (GHz) [default from first freq]: ")
    kappa_input = input("  kappa (GHz) [default 0.01]: ")
    beta_input = input("  beta (GHz) [default 0.01]: ")
    
    cavity_params = {
        'wc': float(wc_input) if wc_input else peak_df['Freq1(ГГц)'].mean(),
        'kappa': float(kappa_input) if kappa_input else 0.01,
        'beta': float(beta_input) if beta_input else 0.01
    }
    
    print(f"\nUsing cavity parameters:")
    print(f"  wc = {cavity_params['wc']:.6f} GHz")
    print(f"  kappa = {cavity_params['kappa']:.6f} GHz")
    print(f"  beta = {cavity_params['beta']:.6f} GHz")
    
    # Построить все графики параметров
    plot_all_parameters(results_dir, peak_df, coupling_df, cavity_params)
    
    # Построить сравнение с моделью
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    data_file = input("\nEnter path to original data file: ")
    
    if data_file and os.path.exists(data_file):
        plot_model_comparison(results_dir, data_file, coupling_df, cavity_params)
    else:
        print("Skipping model comparison (file not found or not provided)")
    
    print("\n" + "=" * 70)
    print("PLOTTING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_manual_results.py <results_directory>")
        print("\nExample:")
        print("  python plot_manual_results.py results/manual_peak_selection_20251113_120000")
        
        # Попробовать найти последнюю директорию автоматически
        results_base = "results"
        if os.path.exists(results_base):
            dirs = [d for d in os.listdir(results_base) if d.startswith("manual_peak_selection_")]
            if dirs:
                dirs.sort(reverse=True)
                latest = os.path.join(results_base, dirs[0])
                print(f"\nLatest results directory found: {latest}")
                use_latest = input("Use this directory? (y/n): ")
                if use_latest.lower() == 'y':
                    main(latest)
                else:
                    sys.exit(1)
            else:
                print("\nNo manual peak selection results found in results/")
                sys.exit(1)
        else:
            sys.exit(1)
    else:
        results_dir = sys.argv[1]
        main(results_dir)
