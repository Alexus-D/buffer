"""
Интерактивное сравнение модели anticrossing с экспериментальными данными

Показывает 3 графика:
1. Экспериментальные данные |S|
2. Модель |S| с параметрами из ползунков
3. Разница (ошибка) между моделью и экспериментом

Управление: ползунки для всех параметров модели
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import models
import data_io

# =============================================================================
# НАСТРОЙКИ (РЕДАКТИРУЙТЕ ЗДЕСЬ)
# =============================================================================

# Путь к файлу с данными
DATA_FILE = r"data\Сфера\CoherentCoupling_S21.txt"

# Диапазоны изменения параметров для ползунков
# Формат: (min, max, initial_value)
PARAM_RANGES = {
    'wc':    (3.0, 4.0, 3.645),      # ГГц - частота резонатора
    'kappa': (0.001, 0.5, 0.294),    # ГГц - ширина резонатора
    'beta':  (0.001, 0.5, 0.327),    # ГГц - параметр связи резонатора
    'J':     (0.001, 0.5, 0.500),    # ГГц - константа связи
    'gamma': (0.001, 0.5, 0.020),    # ГГц - затухание магнона (intrinsic)
    'alpha': (0.0, 0.5, 0.396),      # ГГц - затухание магнона (Gilbert)
    'H_res': (2800.0, 3100.0, 2961.0),  # Э - поле резонанса (где wm = wc)
    'background': (0.0, 1.0, 0.0),   # Фоновые потери (вычитается из |S|)
}

# Gamma вычисляется автоматически: Gamma = sqrt(kappa * gamma)

# Файл для сохранения параметров
PARAMS_SAVE_FILE = "interactive_parameters.txt"

# Физические константы (обычно не меняются)
GAMMA_G = 2.8e-3  # ГГц/Э - гиромагнитное отношение
H0 = 0.0          # Э - поле анизотропии
W0 = 0.0          # ГГц - частота при H=0

# Подвыборка данных для ускорения отрисовки
FREQ_STEP = 5     # Каждая N-я точка по частоте
FIELD_STEP = 1    # Каждая N-я точка по полю

# =============================================================================
# ЗАГРУЗКА ДАННЫХ
# =============================================================================

print("Loading data...")
data = data_io.load_s_parameter_data(DATA_FILE)
freq_full = data['freq']
field_full = data['field']
s_param_full = data['s_param']

# Подвыборка для ускорения
freq = freq_full[::FREQ_STEP]
field = field_full[::FIELD_STEP]
s_exp = s_param_full[::FIELD_STEP, ::FREQ_STEP]
s_exp_mag = np.abs(s_exp)

print(f"Data loaded: {len(field)} fields x {len(freq)} frequencies")
print(f"Field range: {field[0]:.1f} - {field[-1]:.1f} Э")
print(f"Frequency range: {freq[0]:.3f} - {freq[-1]:.3f} ГГц")

# Создать сетки для contourf
FREQ_GRID, FIELD_GRID = np.meshgrid(freq, field)

# =============================================================================
# ФУНКЦИЯ ВЫЧИСЛЕНИЯ МОДЕЛИ
# =============================================================================

def compute_model(wc, kappa, beta, J, gamma, alpha, H_res, background):
    """
    Вычислить модель для всех полей и частот
    
    Parameters:
    -----------
    H_res : float
        Поле резонанса (Э), при котором wm = wc
        Используется для вычисления H0: wc = w0 + gamma_g * (H_res - H0)
    background : float
        Фоновые потери (вычитается из |S|)
    
    Note: Gamma вычисляется автоматически как sqrt(kappa * gamma)
    
    Returns:
    --------
    s_model_mag : array
        Модуль S-параметра модели [n_fields, n_freqs]
    """
    # Вычислить H0 из условия резонанса: wc = w0 + gamma_g * (H_res - H0)
    # H0 = H_res - (wc - w0) / gamma_g
    H0_calc = H_res - (wc - W0) / GAMMA_G
    
    # Вычислить Gamma автоматически
    Gamma = np.sqrt(kappa * gamma)
    
    params = {
        'wc': wc,
        'kappa': kappa,
        'beta': beta,
        'J': J,
        'Gamma': Gamma,
        'gamma': gamma,
        'alpha': alpha,
        'gamma_g': GAMMA_G,
        'H0': H0_calc,
        'w0': W0
    }
    
    s_model = np.zeros((len(field), len(freq)), dtype=complex)
    for i, H in enumerate(field):
        s_model[i, :] = models.anticrossing_one_mode_model(freq, H, params)
    
    # Вычесть фоновые потери
    return np.abs(s_model) - background

# =============================================================================
# СОЗДАНИЕ ИНТЕРАКТИВНЫХ ГРАФИКОВ
# =============================================================================

print("Creating interactive plots...")

# Создать фигуру с 3 графиками
fig = plt.figure(figsize=(18, 6))

# Подграфики для данных
ax1 = plt.subplot(1, 3, 1)
ax2 = plt.subplot(1, 3, 2)
ax3 = plt.subplot(1, 3, 3)

# Настроить расположение для ползунков снизу
plt.subplots_adjust(left=0.05, bottom=0.43, right=0.98, top=0.93, wspace=0.25)

# Начальные значения параметров
initial_params = {key: val[2] for key, val in PARAM_RANGES.items()}

# Вычислить начальную модель
s_model_mag = compute_model(**initial_params)
error = s_model_mag - s_exp_mag

# Общие пределы для colorbar
vmin_data = s_exp_mag.min()
vmax_data = s_exp_mag.max()
vmin_error = -0.2
vmax_error = 0.2

# График 1: Экспериментальные данные
im1 = ax1.contourf(FREQ_GRID, FIELD_GRID, s_exp_mag, levels=50, cmap='viridis', vmin=vmin_data, vmax=vmax_data)
ax1.set_xlabel('Frequency (GHz)', fontsize=10)
ax1.set_ylabel('Field (Oe)', fontsize=10)
ax1.set_title('Experimental |S|', fontsize=10, fontweight='bold')
cbar1 = plt.colorbar(im1, ax=ax1)
cbar1.set_label('|S|', fontsize=9)

# График 2: Модель
im2 = ax2.contourf(FREQ_GRID, FIELD_GRID, s_model_mag, levels=50, cmap='viridis', vmin=vmin_data, vmax=vmax_data)
ax2.set_xlabel('Frequency (GHz)', fontsize=10)
ax2.set_ylabel('Field (Oe)', fontsize=10)
ax2.set_title('Model |S|', fontsize=10, fontweight='bold')
cbar2 = plt.colorbar(im2, ax=ax2)
cbar2.set_label('|S|', fontsize=9)
cbar2_ref = [cbar2]  # Сохранить ссылку на colorbar для обновления

# График 3: Разница (ошибка)
# Симметричные пределы относительно нуля
error_max = max(abs(error.min()), abs(error.max()))
im3 = ax3.contourf(FREQ_GRID, FIELD_GRID, error, levels=50, cmap='RdBu_r', vmin=-error_max, vmax=error_max)
ax3.set_xlabel('Frequency (GHz)', fontsize=10)
ax3.set_ylabel('Field (Oe)', fontsize=10)
ax3.set_title('Error (Model - Exp)', fontsize=10, fontweight='bold')
cbar3 = plt.colorbar(im3, ax=ax3)
cbar3.set_label('Error', fontsize=10)
cbar3_ref = [cbar3]  # Сохранить ссылку на colorbar для обновления

# Добавить текст с RMS ошибкой
rms_text = ax3.text(0.02, 0.98, f'RMS: {np.sqrt(np.mean(error**2)):.4f}',
                    transform=ax3.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# =============================================================================
# СОЗДАНИЕ ПОЛЗУНКОВ
# =============================================================================

# Позиции ползунков (9 параметров вертикально)
slider_height = 0.015
slider_spacing = 0.024
slider_width = 0.80
slider_left = 0.08
slider_bottom = 0.02

slider_axes = []
sliders = []

param_names = ['wc', 'kappa', 'beta', 'J', 'gamma', 'alpha', 'H_res', 'background']
param_labels = ['ωc (GHz)', 'κ (GHz)', 'β (GHz)', 'J (GHz)', 'γ (GHz)', 'α (GHz)', 'H_res (Oe)', 'Background']

for i, (param, label) in enumerate(zip(param_names, param_labels)):
    # Ползунки идут снизу вверх
    y = slider_bottom + i * slider_spacing
    
    ax_slider = plt.axes([slider_left, y, slider_width, slider_height])
    vmin, vmax, vinit = PARAM_RANGES[param]
    
    slider = Slider(
        ax=ax_slider,
        label=label,
        valmin=vmin,
        valmax=vmax,
        valinit=vinit,
        valstep=(vmax - vmin) / 1000  # 1000 шагов
    )
    
    slider_axes.append(ax_slider)
    sliders.append(slider)

# Словарь ползунков для удобного доступа
slider_dict = dict(zip(param_names, sliders))

# Текстовое поле для отображения вычисленного значения Gamma
gamma_text_ax = plt.axes([slider_left, slider_bottom + len(param_names) * slider_spacing, slider_width, slider_height])
gamma_text_ax.axis('off')
initial_gamma_val = np.sqrt(initial_params['kappa'] * initial_params['gamma'])
gamma_text = gamma_text_ax.text(0.02, 0.5, f'Γ (GHz): {initial_gamma_val:.6f} [auto: √(κ×γ)]', 
                                fontsize=9, verticalalignment='center',
                                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# =============================================================================
# ФУНКЦИЯ ОБНОВЛЕНИЯ ГРАФИКОВ
# =============================================================================

def update(val):
    """Обновить графики при изменении ползунков"""
    # Получить текущие значения параметров
    params = {name: slider_dict[name].val for name in param_names}
    
    # Вычислить Gamma автоматически
    Gamma_calc = np.sqrt(params['kappa'] * params['gamma'])
    
    # Обновить текст Gamma
    gamma_text.set_text(f'Γ (GHz): {Gamma_calc:.6f} [auto: √(κ×γ)]')
    
    # Вычислить новую модель
    s_model_mag = compute_model(**params)
    error = s_model_mag - s_exp_mag
    
    # Удалить старые colorbars
    if cbar2_ref[0] is not None:
        cbar2_ref[0].remove()
    if cbar3_ref[0] is not None:
        cbar3_ref[0].remove()
    
    # Очистить графики
    ax2.clear()
    ax3.clear()
    
    # Обновить график модели
    # Динамические пределы для модели
    vmin_model = min(s_exp_mag.min(), s_model_mag.min())
    vmax_model = max(s_exp_mag.max(), s_model_mag.max())
    
    im2 = ax2.contourf(FREQ_GRID, FIELD_GRID, s_model_mag, levels=50, cmap='viridis', vmin=vmin_model, vmax=vmax_model)
    ax2.set_xlabel('Frequency (GHz)', fontsize=10)
    ax2.set_ylabel('Field (Oe)', fontsize=10)
    ax2.set_title('Model |S|', fontsize=10, fontweight='bold')
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('|S|', fontsize=9)
    cbar2_ref[0] = cbar2  # Обновить ссылку
    
    # Обновить график ошибки с симметричными пределами относительно нуля
    error_max = max(abs(error.min()), abs(error.max()))
    im3 = ax3.contourf(FREQ_GRID, FIELD_GRID, error, levels=50, cmap='RdBu_r', 
                       vmin=-error_max, vmax=error_max)
    ax3.set_xlabel('Frequency (GHz)', fontsize=10)
    ax3.set_ylabel('Field (Oe)', fontsize=10)
    ax3.set_title('Error (Model - Exp)', fontsize=10, fontweight='bold')
    cbar3 = plt.colorbar(im3, ax=ax3)
    cbar3.set_label('Error', fontsize=10)
    cbar3_ref[0] = cbar3  # Обновить ссылку
    
    # Обновить текст RMS
    rms = np.sqrt(np.mean(error**2))
    ax3.text(0.02, 0.98, f'RMS: {rms:.4f}',
             transform=ax3.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Перерисовать
    fig.canvas.draw_idle()

# Подключить обновление к ползункам
for slider in sliders:
    slider.on_changed(update)

# =============================================================================
# КНОПКА СБРОСА
# =============================================================================

from matplotlib.widgets import Button

# Кнопка сброса
reset_ax = plt.axes([0.35, 0.26, 0.12, 0.03])
button_reset = Button(reset_ax, 'Reset', color='lightgoldenrodyellow', hovercolor='0.975')

def reset(event):
    """Сбросить все ползунки к начальным значениям"""
    for param, slider in slider_dict.items():
        slider.reset()

button_reset.on_clicked(reset)

# Кнопка сохранения параметров
save_ax = plt.axes([0.53, 0.26, 0.12, 0.03])
button_save = Button(save_ax, 'Save Params', color='lightblue', hovercolor='0.975')

def save_params(event):
    """Сохранить текущие значения параметров и графики в файл"""
    from datetime import datetime
    import os
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    timestamp_file = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Создать папку для сохранения если нужно
    save_dir = "interactive_results"
    os.makedirs(save_dir, exist_ok=True)
    
    # Вычислить Gamma
    Gamma_calc = np.sqrt(slider_dict['kappa'].val * slider_dict['gamma'].val)
    
    # Сохранить параметры
    param_file = os.path.join(save_dir, f"params_{timestamp_file}.txt")
    with open(param_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("INTERACTIVE MODEL PARAMETERS\n")
        f.write("=" * 70 + "\n")
        f.write(f"Saved: {timestamp}\n")
        f.write("\n")
        
        for param, label in zip(param_names, param_labels):
            value = slider_dict[param].val
            f.write(f"{label:20s}: {value:.6f}\n")
        
        # Добавить вычисленное значение Gamma
        f.write(f"{'Γ (GHz) [auto]':20s}: {Gamma_calc:.6f}\n")
        f.write("\n")
        f.write("=" * 70 + "\n")
    
    # Сохранить графики
    plot_file = os.path.join(save_dir, f"comparison_{timestamp_file}.png")
    fig.savefig(plot_file, dpi=300, bbox_inches='tight')
    
    # Также сохранить в основной файл для обратной совместимости
    with open(PARAMS_SAVE_FILE, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("INTERACTIVE MODEL PARAMETERS\n")
        f.write("=" * 70 + "\n")
        f.write(f"Saved: {timestamp}\n")
        f.write("\n")
        
        for param, label in zip(param_names, param_labels):
            value = slider_dict[param].val
            f.write(f"{label:20s}: {value:.6f}\n")
        
        # Добавить вычисленное значение Gamma
        f.write(f"{'Γ (GHz) [auto]':20s}: {Gamma_calc:.6f}\n")
        f.write("\n")
        f.write("=" * 70 + "\n")
    
    print(f"\nParameters saved to: {param_file}")
    print(f"Plot saved to: {plot_file}")
    print("Current values:")
    for param, label in zip(param_names, param_labels):
        value = slider_dict[param].val
        print(f"  {label:20s}: {value:.6f}")
    print(f"  {'Γ (GHz) [auto]':20s}: {Gamma_calc:.6f}")

button_save.on_clicked(save_params)

# Кнопка центрирования диапазонов
center_ax = plt.axes([0.35, 0.30, 0.12, 0.03])
button_center = Button(center_ax, 'Center Range', color='lightgreen', hovercolor='0.975')

def center_ranges(event):
    """Сдвинуть диапазоны ползунков так, чтобы текущее значение было в центре"""
    for param_name, slider in slider_dict.items():
        current_val = slider.val
        current_min = slider.valmin
        current_max = slider.valmax
        range_width = current_max - current_min
        
        # Новые границы: текущее значение ± половина диапазона
        new_min = current_val - range_width / 2
        new_max = current_val + range_width / 2
        
        # Ограничить минимум нулём
        if new_min < 0:
            new_min = 0.0
            new_max = range_width
        
        # Обновить границы ползунка
        slider.valmin = new_min
        slider.valmax = new_max
        slider.ax.set_xlim(new_min, new_max)
        
    print("\nRanges centered around current values")
    fig.canvas.draw_idle()

button_center.on_clicked(center_ranges)

# Кнопка сужения диапазонов
narrow_ax = plt.axes([0.53, 0.30, 0.12, 0.03])
button_narrow = Button(narrow_ax, 'Narrow ×5', color='lightsalmon', hovercolor='0.975')

def narrow_ranges(event):
    """Уменьшить диапазоны ползунков в 5 раз вокруг текущего значения"""
    for param_name, slider in slider_dict.items():
        current_val = slider.val
        current_min = slider.valmin
        current_max = slider.valmax
        range_width = current_max - current_min
        
        # Новый диапазон: в 5 раз меньше
        new_range_width = range_width / 5
        new_min = current_val - new_range_width / 2
        new_max = current_val + new_range_width / 2
        
        # Ограничить минимум нулём
        if new_min < 0:
            new_min = 0.0
            new_max = new_range_width
        
        # Обновить границы ползунка
        slider.valmin = new_min
        slider.valmax = new_max
        slider.ax.set_xlim(new_min, new_max)
        
    print("\nRanges narrowed by factor of 5")
    fig.canvas.draw_idle()

button_narrow.on_clicked(narrow_ranges)

# =============================================================================
# ВЫВОД ИНСТРУКЦИЙ И ПОКАЗ ГРАФИКОВ
# =============================================================================

print("\n" + "=" * 70)
print("INTERACTIVE MODEL COMPARISON")
print("=" * 70)
print("\nInstructions:")
print("  - Use sliders to adjust model parameters")
print("  - Click 'Reset' to restore initial values")
print("  - Close window to exit")
print("\nCurrent parameters:")
for param, label in zip(param_names, param_labels):
    print(f"  {label:12s}: {initial_params[param]:.6f}")
print("\n" + "=" * 70)

plt.suptitle('Anticrossing Model vs Experimental Data - Interactive Comparison', 
             fontsize=11, fontweight='bold', y=0.99)
plt.show()

print("\nDone!")
