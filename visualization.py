"""
Функции визуализации данных и результатов

Содержит функции для построения графиков и контурных карт
"""

import matplotlib.pyplot as plt
import numpy as np

import models

# =============================================================================
# КОНТУРНЫЕ КАРТЫ
# =============================================================================

def plot_contour_map(data, title='', save_path=None, show=False):
    """
    Построить контурную карту S-параметра
    
    Parameters:
    -----------
    data : dict
        Словарь с данными:
        - 'freq': массив частот (ГГц)
        - 'field': массив полей (Э)
        - 's_param': 2D массив S-параметров
        - 's_type': тип S-параметра ('S21' или 'S12')
    title : str
        Заголовок графика
    save_path : str or None
        Путь для сохранения графика
    show : bool
        Если True, отобразить график на экране (по умолчанию False)
    """
    # Извлечение данных
    freq = data['freq']
    field = data['field']
    s_param = data['s_param']
    s_type = data.get('s_type', 'S21')
    
    # Вычисление амплитуды в dB:
    s_amplitude_db = models.convert_linear_to_dB(np.abs(s_param))
    
    # Создание сетки для контурной карты
    freq_grid, field_grid = np.meshgrid(freq, field)
    
    # Создание фигуры
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Построение контурной карты
    contour = ax.contourf(freq_grid, field_grid, s_amplitude_db, 
                          levels=50, cmap='viridis')
    
    # Добавление цветовой шкалы
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label(f'|{s_type}| (dB)', rotation=270, labelpad=20)
    
    # Настройка осей
    ax.set_xlabel('Частота (ГГц)', fontsize=12)
    ax.set_ylabel('Магнитное поле (Э)', fontsize=12)
    
    # Заголовок
    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title(f'Контурная карта {s_type}', fontsize=14)
    
    # Сетка
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Плотная компоновка
    plt.tight_layout()
    
    # Сохранение графика
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  График сохранен: {save_path}")
    
    # Отображение на экране
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_comparison_contours(experimental_data, fitted_data, save_path=None, show=False):
    """
    Построить сравнение экспериментальных и подогнанных контурных карт
    
    Parameters:
    -----------
    experimental_data : dict
        Экспериментальные данные:
        - 'freq': массив частот (ГГц)
        - 'field': массив полей (Э)
        - 's_param': 2D массив S-параметров
        - 's_type': тип S-параметра ('S21' или 'S12')
    fitted_data : dict
        Подогнанные данные (та же структура)
    save_path : str or None
        Путь для сохранения графика
    show : bool
        Если True, отобразить график на экране (по умолчанию False)
        
    Создает три контурные карты:
    1. Экспериментальные данные
    2. Подогнанные данные
    3. Разница (остатки)
    """
    # Извлечение данных
    freq = experimental_data['freq']
    field = experimental_data['field']
    s_exp = experimental_data['s_param']
    s_fit = fitted_data['s_param']
    s_type = experimental_data.get('s_type', 'S21')
    
    # Вычисление амплитуд в dB
    s_exp_db = models.convert_linear_to_dB(np.abs(s_exp))
    s_fit_db = models.convert_linear_to_dB(np.abs(s_fit))
    s_residual_db = s_exp_db - s_fit_db
    
    # Создание сетки для контурных карт
    freq_grid, field_grid = np.meshgrid(freq, field)
    
    # Создание фигуры с тремя подграфиками
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Определение общих пределов цветовой шкалы для первых двух графиков
    vmin = min(s_exp_db.min(), s_fit_db.min())
    vmax = max(s_exp_db.max(), s_fit_db.max())
    
    # График 1: Экспериментальные данные
    contour1 = axes[0].contourf(freq_grid, field_grid, s_exp_db, 
                                 levels=50, cmap='viridis', vmin=vmin, vmax=vmax)
    cbar1 = plt.colorbar(contour1, ax=axes[0])
    cbar1.set_label(f'|{s_type}| (dB)', rotation=270, labelpad=20)
    axes[0].set_xlabel('Частота (ГГц)', fontsize=11)
    axes[0].set_ylabel('Магнитное поле (Э)', fontsize=11)
    axes[0].set_title('Экспериментальные данные', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3, linestyle='--')
    
    # График 2: Подогнанные данные
    contour2 = axes[1].contourf(freq_grid, field_grid, s_fit_db, 
                                 levels=50, cmap='viridis', vmin=vmin, vmax=vmax)
    cbar2 = plt.colorbar(contour2, ax=axes[1])
    cbar2.set_label(f'|{s_type}| (dB)', rotation=270, labelpad=20)
    axes[1].set_xlabel('Частота (ГГц)', fontsize=11)
    axes[1].set_ylabel('Магнитное поле (Э)', fontsize=11)
    axes[1].set_title('Подогнанные данные', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, linestyle='--')
    
    # График 3: Разница (остатки)
    # Используем симметричную шкалу для остатков
    residual_max = max(abs(s_residual_db.min()), abs(s_residual_db.max()))
    contour3 = axes[2].contourf(freq_grid, field_grid, s_residual_db, 
                                 levels=50, cmap='RdBu_r', 
                                 vmin=-residual_max, vmax=residual_max)
    cbar3 = plt.colorbar(contour3, ax=axes[2])
    cbar3.set_label('Разница (dB)', rotation=270, labelpad=20)
    axes[2].set_xlabel('Частота (ГГц)', fontsize=11)
    axes[2].set_ylabel('Магнитное поле (Э)', fontsize=11)
    axes[2].set_title('Остатки (эксп. - фит)', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3, linestyle='--')
    
    # Общий заголовок
    fig.suptitle('Сравнение экспериментальных и подогнанных данных', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    # Плотная компоновка
    plt.tight_layout()
    
    # Сохранение графика
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  График сравнения сохранен: {save_path}")
    
    # Отображение на экране
    if show:
        plt.show()
    else:
        plt.close(fig)
    # TODO: Реализовать сравнение контурных карт
    pass


def plot_cavity_fit(data, cavity_params, save_path=None):
    """
    Построить график подгонки резонатора
    
    Parameters:
    -----------
    data : dict
        Экспериментальные данные
    cavity_params : dict
        Параметры резонатора (wc, kappa, beta)
    save_path : str or None
        Путь для сохранения графика
        
    Заглушка для Этапа 1
    """
    # TODO: Реализовать построение графика подгонки резонатора
    pass


# =============================================================================
# ЛИНЕЙНЫЕ ГРАФИКИ
# =============================================================================

def plot_cross_sections(data, field_values=None, freq_values=None, save_path=None):
    """
    Построить срезы S-параметра при фиксированных полях/частотах
    
    Parameters:
    -----------
    data : dict
        Словарь с данными
    field_values : list or None
        Значения полей для срезов
    freq_values : list or None
        Значения частот для срезов
    save_path : str or None
        Путь для сохранения графика
        
    Заглушка для Этапа 1
    """
    # TODO: Реализовать построение срезов
    pass


def plot_magnon_dispersion(field, params, save_path=None):
    """
    Построить дисперсию магнонных мод
    
    Parameters:
    -----------
    field : array-like
        Массив магнитных полей
    params : dict
        Параметры магнонных мод
    save_path : str or None
        Путь для сохранения графика
        
    Заглушка для Этапа 1
    """
    # TODO: Реализовать дисперсию магнонов
    pass


def plot_anticrossing_diagram(data, fitted_params, save_path=None):
    """
    Построить диаграмму антикроссинга с экспериментальными точками и теорией
    
    Parameters:
    -----------
    data : dict
        Экспериментальные данные
    fitted_params : dict
        Подогнанные параметры
    save_path : str or None
        Путь для сохранения графика
        
    Заглушка для Этапа 1
    """
    # TODO: Реализовать диаграмму антикроссинга
    pass


# =============================================================================
# СТАТИСТИКА И МЕТРИКИ
# =============================================================================

def plot_residuals(experimental, fitted, save_path=None):
    """
    Построить контурные карты остатков (residuals)
    
    Parameters:
    -----------
    experimental : array-like
        Экспериментальные данные
    fitted : array-like
        Подогнанные данные
    save_path : str or None
        Путь для сохранения графика
        
    Заглушка для Этапа 1
    """
    # TODO: Реализовать график остатков
    pass


def plot_fitting_quality(data, metrics, save_path=None):
    """
    Построить график качества подгонки (R², RMSE и т.д.)
    
    Parameters:
    -----------
    data : dict
        Данные подгонки
    metrics : dict
        Метрики качества подгонки
    save_path : str or None
        Путь для сохранения графика
        
    Заглушка для Этапа 1
    """
    # TODO: Реализовать график качества подгонки
    pass


def plot_fitted_parameters_vs_field(fitting_results, save_path=None, show=False):
    """
    Построить графики зависимости подогнанных параметров от магнитного поля
    
    Parameters:
    -----------
    fitting_results : list of dict
        Список результатов фиттинга для каждого поля
        Каждый элемент содержит:
        - 'fitted_params': dict с параметрами (J, gamma, alpha, Gamma, wc, kappa, beta)
        - 'fit_quality': dict с метриками качества (r_squared, rmse)
    save_path : str or None
        Путь для сохранения графика
    show : bool
        Если True, отобразить график на экране (по умолчанию False)
        
    Создает графики:
    - J (когерентная связь) vs поле
    - gamma (внешние потери магнонов) vs поле
    - alpha (внутренние потери магнонов) vs поле
    - Gamma (диссипативная связь) vs поле
    """
    # Извлечение данных из результатов фиттинга
    fields = []
    J_values = []
    gamma_values = []
    alpha_values = []
    Gamma_values = []
    
    for result in fitting_results:
        if result is not None:
            params = result['fitted_params']
            fields.append(params['field'])
            J_values.append(params['J'])
            gamma_values.append(params['gamma'])
            alpha_values.append(params['alpha'])
            Gamma_values.append(params['Gamma'])
    
    # Преобразование в numpy массивы
    fields = np.array(fields)
    J_values = np.array(J_values)
    gamma_values = np.array(gamma_values)
    alpha_values = np.array(alpha_values)
    Gamma_values = np.array(Gamma_values)
    
    # Создание фигуры с 4 подграфиками (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # График 1: J (когерентная связь)
    axes[0, 0].plot(fields, J_values, 'o-', color='#1f77b4', linewidth=2, markersize=4)
    axes[0, 0].set_xlabel('Магнитное поле (Э)', fontsize=11)
    axes[0, 0].set_ylabel('J (ГГц)', fontsize=11)
    axes[0, 0].set_title('Когерентная связь vs поле', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, linestyle='--')
    axes[0, 0].axhline(y=np.mean(J_values), color='red', linestyle='--', 
                       linewidth=1, alpha=0.7, label=f'Среднее: {np.mean(J_values):.6f} ГГц')
    axes[0, 0].legend(fontsize=9)
    
    # График 2: gamma (внешние потери магнонов)
    axes[0, 1].plot(fields, gamma_values, 'o-', color='#ff7f0e', linewidth=2, markersize=4)
    axes[0, 1].set_xlabel('Магнитное поле (Э)', fontsize=11)
    axes[0, 1].set_ylabel('γ (ГГц)', fontsize=11)
    axes[0, 1].set_title('Внешние потери магнонов vs поле', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, linestyle='--')
    axes[0, 1].axhline(y=np.mean(gamma_values), color='red', linestyle='--', 
                       linewidth=1, alpha=0.7, label=f'Среднее: {np.mean(gamma_values):.6f} ГГц')
    axes[0, 1].legend(fontsize=9)
    
    # График 3: alpha (внутренние потери магнонов)
    axes[1, 0].plot(fields, alpha_values, 'o-', color='#2ca02c', linewidth=2, markersize=4)
    axes[1, 0].set_xlabel('Магнитное поле (Э)', fontsize=11)
    axes[1, 0].set_ylabel('α (ГГц)', fontsize=11)
    axes[1, 0].set_title('Внутренние потери магнонов vs поле', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, linestyle='--')
    axes[1, 0].axhline(y=np.mean(alpha_values), color='red', linestyle='--', 
                       linewidth=1, alpha=0.7, label=f'Среднее: {np.mean(alpha_values):.6f} ГГц')
    axes[1, 0].legend(fontsize=9)
    
    # График 4: Gamma (диссипативная связь)
    axes[1, 1].plot(fields, Gamma_values, 'o-', color='#d62728', linewidth=2, markersize=4)
    axes[1, 1].set_xlabel('Магнитное поле (Э)', fontsize=11)
    axes[1, 1].set_ylabel('Γ (ГГц)', fontsize=11)
    axes[1, 1].set_title('Диссипативная связь vs поле', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, linestyle='--')
    axes[1, 1].axhline(y=np.mean(Gamma_values), color='red', linestyle='--', 
                       linewidth=1, alpha=0.7, label=f'Среднее: {np.mean(Gamma_values):.6f} ГГц')
    axes[1, 1].legend(fontsize=9)
    
    # Общий заголовок
    fig.suptitle('Зависимость подогнанных параметров от магнитного поля', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    # Плотная компоновка
    plt.tight_layout()
    
    # Сохранение графика
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  График параметров vs поле сохранен: {save_path}")
    
    # Отображение на экране
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_fitting_errors_vs_field(fitting_results, save_path=None, show=False):
    """
    Построить графики ошибок подгонки vs магнитное поле
    
    Parameters:
    -----------
    fitting_results : list of dict
        Список результатов фиттинга для каждого поля
        Каждый элемент содержит:
        - 'fitted_params': dict с параметрами
        - 'fit_quality': dict с метриками качества (r_squared, rmse)
    save_path : str or None
        Путь для сохранения графика
    show : bool
        Если True, отобразить график на экране (по умолчанию False)
        
    Создает графики:
    - R² (коэффициент детерминации) vs поле
    - RMSE (среднеквадратичная ошибка) vs поле
    - Гистограммы распределения R² и RMSE
    """
    # Извлечение данных из результатов фиттинга
    fields = []
    r_squared_values = []
    rmse_values = []
    
    for result in fitting_results:
        if result is not None:
            params = result['fitted_params']
            quality = result['fit_quality']
            fields.append(params['field'])
            r_squared_values.append(quality['r_squared'])
            rmse_values.append(quality['rmse'])
    
    # Преобразование в numpy массивы
    fields = np.array(fields)
    r_squared_values = np.array(r_squared_values)
    rmse_values = np.array(rmse_values)
    
    # Создание фигуры с 4 подграфиками (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # График 1: R² vs поле
    axes[0, 0].plot(fields, r_squared_values, 'o-', color='#9467bd', linewidth=2, markersize=4)
    axes[0, 0].set_xlabel('Магнитное поле (Э)', fontsize=11)
    axes[0, 0].set_ylabel('R²', fontsize=11)
    axes[0, 0].set_title('Коэффициент детерминации R² vs поле', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, linestyle='--')
    axes[0, 0].axhline(y=np.mean(r_squared_values), color='red', linestyle='--', 
                       linewidth=1, alpha=0.7, label=f'Среднее: {np.mean(r_squared_values):.6f}')
    axes[0, 0].set_ylim([max(0, np.min(r_squared_values) - 0.01), 
                         min(1, np.max(r_squared_values) + 0.01)])
    axes[0, 0].legend(fontsize=9)
    
    # График 2: RMSE vs поле
    axes[0, 1].plot(fields, rmse_values, 'o-', color='#8c564b', linewidth=2, markersize=4)
    axes[0, 1].set_xlabel('Магнитное поле (Э)', fontsize=11)
    axes[0, 1].set_ylabel('RMSE', fontsize=11)
    axes[0, 1].set_title('Среднеквадратичная ошибка vs поле', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, linestyle='--')
    axes[0, 1].axhline(y=np.mean(rmse_values), color='red', linestyle='--', 
                       linewidth=1, alpha=0.7, label=f'Среднее: {np.mean(rmse_values):.6f}')
    axes[0, 1].legend(fontsize=9)
    
    # График 3: Гистограмма R²
    axes[1, 0].hist(r_squared_values, bins=30, color='#9467bd', alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(x=np.mean(r_squared_values), color='red', linestyle='--', 
                       linewidth=2, label=f'Среднее: {np.mean(r_squared_values):.6f}')
    axes[1, 0].axvline(x=np.median(r_squared_values), color='orange', linestyle='--', 
                       linewidth=2, label=f'Медиана: {np.median(r_squared_values):.6f}')
    axes[1, 0].set_xlabel('R²', fontsize=11)
    axes[1, 0].set_ylabel('Количество', fontsize=11)
    axes[1, 0].set_title('Распределение R²', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, linestyle='--', axis='y')
    axes[1, 0].legend(fontsize=9)
    
    # График 4: Гистограмма RMSE
    axes[1, 1].hist(rmse_values, bins=30, color='#8c564b', alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(x=np.mean(rmse_values), color='red', linestyle='--', 
                       linewidth=2, label=f'Среднее: {np.mean(rmse_values):.6f}')
    axes[1, 1].axvline(x=np.median(rmse_values), color='orange', linestyle='--', 
                       linewidth=2, label=f'Медиана: {np.median(rmse_values):.6f}')
    axes[1, 1].set_xlabel('RMSE', fontsize=11)
    axes[1, 1].set_ylabel('Количество', fontsize=11)
    axes[1, 1].set_title('Распределение RMSE', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, linestyle='--', axis='y')
    axes[1, 1].legend(fontsize=9)
    
    # Общий заголовок
    fig.suptitle('Анализ качества подгонки', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    # Плотная компоновка
    plt.tight_layout()
    
    # Сохранение графика
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  График ошибок подгонки сохранен: {save_path}")
    
    # Отображение на экране
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_cavity_fit_cross_section(fitted_spectrum, fit_quality, save_path=None, show=False):
    """
    Построить график спектра резонатора: экспериментальные и фитированные данные
    
    Parameters:
    -----------
    fitted_spectrum : dict
        Результаты фиттинга резонатора:
        - 'freq': массив частот (ГГц)
        - 's_param_avg': усредненные экспериментальные данные
        - 's_param_fitted': фитированные данные
        - 's_type': тип S-параметра ('S21' или 'S12')
    fit_quality : dict
        Метрики качества подгонки (r_squared, rmse)
    save_path : str or None
        Путь для сохранения графика
    show : bool
        Если True, отобразить график на экране
    """
    if fitted_spectrum is None:
        print("  Нет данных для построения графика фиттинга резонатора")
        return
    
    freq = fitted_spectrum['freq']
    s_exp = fitted_spectrum['s_param_avg']
    s_fit = fitted_spectrum['s_param_fitted']
    s_type = fitted_spectrum.get('s_type', 'S21')
    
    # Вычислить амплитуду в dB
    s_exp_db = models.convert_linear_to_dB(s_exp)
    s_fit_db = models.convert_linear_to_dB(s_fit)
    
    # Вычислить остатки
    residuals = s_exp - s_fit
    
    # Создать фигуру с двумя подграфиками
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Верхний график: экспериментальные и фитированные данные
    ax1.plot(freq, s_exp_db, 'o-', label='Экспериментальные данные', 
             color='blue', markersize=4, linewidth=1)
    ax1.plot(freq, s_fit_db, 's-', label='Подогнанные данные', 
             color='red', markersize=3, linewidth=2)
    ax1.set_xlabel('Частота (ГГц)', fontsize=12)
    ax1.set_ylabel(f'|{s_type}| (dB)', fontsize=12)
    ax1.set_title('Фиттинг резонатора (усреднено по полю)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Добавить текст с метриками качества
    quality_text = f'R² = {fit_quality["r_squared"]:.6f}\nRMSE = {fit_quality["rmse"]:.6f}'
    ax1.text(0.02, 0.98, quality_text, transform=ax1.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Нижний график: остатки (residuals)
    ax2.plot(freq, residuals, 'o-', color='green', markersize=4, linewidth=1)
    ax2.axhline(0, color='black', linestyle='--', linewidth=1)
    ax2.set_xlabel('Частота (ГГц)', fontsize=12)
    ax2.set_ylabel('Остатки (линейные единицы)', fontsize=12)
    ax2.set_title('Остатки подгонки', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Сохранение графика
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  График фиттинга резонатора сохранен: {save_path}")
    
    # Отображение на экране
    if show:
        plt.show()
    else:
        plt.close(fig)


# =============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =============================================================================

def configure_plot_style():
    """
    Настроить стиль графиков matplotlib
    
    Заглушка для Этапа 1
    """
    # TODO: Реализовать настройку стиля
    pass


def save_figure(fig, filepath, dpi=300, format='png'):
    """
    Сохранить фигуру matplotlib
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        Фигура для сохранения
    filepath : str
        Путь к файлу
    dpi : int
        Разрешение
    format : str
        Формат файла
        
    Заглушка для Этапа 1
    """
    # TODO: Реализовать сохранение фигуры
    pass


# =============================================================================
# ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ ОТСЛЕЖИВАНИЯ ПИКОВ
# =============================================================================

def plot_eigenfrequencies_vs_field(results, save_path=None, show=False):
    """
    Построить график собственных частот системы vs магнитное поле
    
    Parameters:
    -----------
    results : list of dict
        Результаты отслеживания пиков:
        - 'field': значение поля (Э)
        - 'f1', 'f2': собственные частоты двух мод (ГГц)
    save_path : str or None
        Путь для сохранения графика
    show : bool
        Если True, отобразить график на экране
    """
    # Извлечение данных
    fields = [r['field'] for r in results]
    f1_values = [r['f1'] for r in results]
    f2_values = [r['f2'] for r in results]
    
    # Создание фигуры
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Построение графиков
    ax.plot(fields, f1_values, 'o-', color='blue', linewidth=2, 
            markersize=6, label='Мода 1', alpha=0.8)
    ax.plot(fields, f2_values, 's-', color='red', linewidth=2, 
            markersize=6, label='Мода 2', alpha=0.8)
    
    # Настройка осей
    ax.set_xlabel('Магнитное поле (Э)', fontsize=12)
    ax.set_ylabel('Собственная частота (ГГц)', fontsize=12)
    ax.set_title('Собственные частоты системы vs магнитное поле', fontsize=14)
    
    # Сетка и легенда
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='best')
    
    # Плотная компоновка
    plt.tight_layout()
    
    # Сохранение графика
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  График сохранен: {save_path}")
    
    # Отображение на экране
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_peak_widths_vs_field(results, save_path=None, show=False):
    """
    Построить график ширин пиков vs магнитное поле
    
    Parameters:
    -----------
    results : list of dict
        Результаты отслеживания пиков:
        - 'field': значение поля (Э)
        - 'w1', 'w2': ширины пиков (ГГц)
    save_path : str or None
        Путь для сохранения графика
    show : bool
        Если True, отобразить график на экране
    """
    # Извлечение данных
    fields = [r['field'] for r in results]
    w1_values = [r['w1'] for r in results]
    w2_values = [r['w2'] for r in results]
    
    # Создание фигуры
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Построение графиков
    ax.plot(fields, w1_values, 'o-', color='blue', linewidth=2, 
            markersize=6, label='Ширина моды 1', alpha=0.8)
    ax.plot(fields, w2_values, 's-', color='red', linewidth=2, 
            markersize=6, label='Ширина моды 2', alpha=0.8)
    
    # Настройка осей
    ax.set_xlabel('Магнитное поле (Э)', fontsize=12)
    ax.set_ylabel('Ширина пика (ГГц)', fontsize=12)
    ax.set_title('Ширины пиков vs магнитное поле', fontsize=14)
    
    # Сетка и легенда
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='best')
    
    # Плотная компоновка
    plt.tight_layout()
    
    # Сохранение графика
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  График сохранен: {save_path}")
    
    # Отображение на экране
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_peak_amplitudes_vs_field(results, save_path=None, show=False):
    """
    Построить график амплитуд пиков vs магнитное поле
    
    Parameters:
    -----------
    results : list of dict
        Результаты отслеживания пиков:
        - 'field': значение поля (Э)
        - 'a1', 'a2': амплитуды пиков
    save_path : str or None
        Путь для сохранения графика
    show : bool
        Если True, отобразить график на экране
    """
    # Извлечение данных
    fields = [r['field'] for r in results]
    a1_values = [r['a1'] for r in results]
    a2_values = [r['a2'] for r in results]
    
    # Создание фигуры
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Построение графиков
    ax.plot(fields, a1_values, 'o-', color='blue', linewidth=2, 
            markersize=6, label='Амплитуда моды 1', alpha=0.8)
    ax.plot(fields, a2_values, 's-', color='red', linewidth=2, 
            markersize=6, label='Амплитуда моды 2', alpha=0.8)
    
    # Настройка осей
    ax.set_xlabel('Магнитное поле (Э)', fontsize=12)
    ax.set_ylabel('Амплитуда пика (провала)', fontsize=12)
    ax.set_title('Амплитуды пиков vs магнитное поле', fontsize=14)
    
    # Сетка и легенда
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='best')
    
    # Плотная компоновка
    plt.tight_layout()
    
    # Сохранение графика
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  График сохранен: {save_path}")
    
    # Отображение на экране
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_peak_fit_quality_vs_field(results, save_path=None, show=False):
    """
    Построить график качества подгонки vs магнитное поле
    
    Parameters:
    -----------
    results : list of dict
        Результаты отслеживания пиков:
        - 'field': значение поля (Э)
        - 'r_squared': коэффициент детерминации
        - 'rmse': среднеквадратичная ошибка
    save_path : str or None
        Путь для сохранения графика
    show : bool
        Если True, отобразить график на экране
    """
    # Извлечение данных
    fields = [r['field'] for r in results]
    r2_values = [r['r_squared'] for r in results]
    rmse_values = [r['rmse'] for r in results]
    
    # Создание фигуры с двумя субплотами
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # График R²
    ax1.plot(fields, r2_values, 'o-', color='green', linewidth=2, 
            markersize=6, alpha=0.8)
    ax1.set_ylabel('R² (коэффициент детерминации)', fontsize=12)
    ax1.set_title('Качество подгонки vs магнитное поле', fontsize=14)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.axhline(y=0.95, color='red', linestyle='--', linewidth=1, alpha=0.5, label='R² = 0.95')
    ax1.legend(fontsize=10, loc='best')
    
    # График RMSE
    ax2.plot(fields, rmse_values, 's-', color='orange', linewidth=2, 
            markersize=6, alpha=0.8)
    ax2.set_xlabel('Магнитное поле (Э)', fontsize=12)
    ax2.set_ylabel('RMSE (среднеквадратичная ошибка)', fontsize=12)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # Плотная компоновка
    plt.tight_layout()
    
    # Сохранение графика
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  График сохранен: {save_path}")
    
    # Отображение на экране
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_peak_tracking_summary(results, save_path=None, show=False):
    """
    Построить сводный график всех параметров пиков
    
    Parameters:
    -----------
    results : list of dict
        Результаты отслеживания пиков
    save_path : str or None
        Путь для сохранения графика
    show : bool
        Если True, отобразить график на экране
    """
    # Извлечение данных
    fields = [r['field'] for r in results]
    f1_values = [r['f1'] for r in results]
    f2_values = [r['f2'] for r in results]
    w1_values = [r['w1'] for r in results]
    w2_values = [r['w2'] for r in results]
    a1_values = [r['a1'] for r in results]
    a2_values = [r['a2'] for r in results]
    r2_values = [r['r_squared'] for r in results]
    
    # Создание фигуры с 4 субплотами
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Собственные частоты
    ax1.plot(fields, f1_values, 'o-', color='blue', linewidth=2, 
            markersize=5, label='Мода 1', alpha=0.8)
    ax1.plot(fields, f2_values, 's-', color='red', linewidth=2, 
            markersize=5, label='Мода 2', alpha=0.8)
    ax1.set_ylabel('Собственная частота (ГГц)', fontsize=11)
    ax1.set_title('Собственные частоты', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=10, loc='best')
    
    # 2. Ширины пиков
    ax2.plot(fields, w1_values, 'o-', color='blue', linewidth=2, 
            markersize=5, label='Ширина 1', alpha=0.8)
    ax2.plot(fields, w2_values, 's-', color='red', linewidth=2, 
            markersize=5, label='Ширина 2', alpha=0.8)
    ax2.set_ylabel('Ширина пика (ГГц)', fontsize=11)
    ax2.set_title('Ширины пиков', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(fontsize=10, loc='best')
    
    # 3. Амплитуды пиков
    ax3.plot(fields, a1_values, 'o-', color='blue', linewidth=2, 
            markersize=5, label='Амплитуда 1', alpha=0.8)
    ax3.plot(fields, a2_values, 's-', color='red', linewidth=2, 
            markersize=5, label='Амплитуда 2', alpha=0.8)
    ax3.set_xlabel('Магнитное поле (Э)', fontsize=11)
    ax3.set_ylabel('Амплитуда пика', fontsize=11)
    ax3.set_title('Амплитуды пиков', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.legend(fontsize=10, loc='best')
    
    # 4. Качество подгонки
    ax4.plot(fields, r2_values, 'o-', color='green', linewidth=2, 
            markersize=5, alpha=0.8)
    ax4.set_xlabel('Магнитное поле (Э)', fontsize=11)
    ax4.set_ylabel('R² (коэффициент детерминации)', fontsize=11)
    ax4.set_title('Качество подгонки', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.axhline(y=0.95, color='red', linestyle='--', linewidth=1, alpha=0.5)
    
    # Общий заголовок
    fig.suptitle('Сводка отслеживания пиков', fontsize=14, fontweight='bold', y=0.995)
    
    # Плотная компоновка
    plt.tight_layout()
    
    # Сохранение графика
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  График сохранен: {save_path}")
    
    # Отображение на экране
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_example_peak_fits(data, results, field_indices=None, 
                           save_path=None, show=False):
    """
    Построить примеры аппроксимации спектров для нескольких полей
    
    Parameters:
    -----------
    data : dict
        Данные S-параметров:
        - 'freq': массив частот (ГГц)
        - 'field': массив полей (Э)
        - 's_param': 2D массив S-параметров
    results : list of dict
        Результаты отслеживания пиков
    field_indices : list of int or None
        Индексы полей для отображения (по умолчанию: начало, середина, конец)
    save_path : str or None
        Путь для сохранения графика
    show : bool
        Если True, отобразить график на экране
    """
    import peak_tracking
    
    # Если индексы не указаны, используем начало, середину и конец
    if field_indices is None:
        field_indices = [0, len(results)//2, len(results)-1]
    
    freq = data['freq']
    field = data['field']
    s_param = data['s_param']
    
    # Количество примеров
    n_examples = len(field_indices)
    
    # Создание фигуры
    fig, axes = plt.subplots(n_examples, 1, figsize=(12, 4*n_examples))
    if n_examples == 1:
        axes = [axes]
    
    for i, idx in enumerate(field_indices):
        if idx >= len(results):
            idx = len(results) - 1
        
        result = results[idx]
        field_value = result['field']
        
        # Найти индекс поля в данных
        field_idx = np.argmin(np.abs(field - field_value))
        
        # Экспериментальный спектр
        spectrum_exp = np.abs(s_param[field_idx, :])
        
        # Восстановленный спектр из параметров
        spectrum_fit = peak_tracking.two_lorentzians_dip(
            freq,
            result['f1'], result['w1'], result['a1'],
            result['f2'], result['w2'], result['a2'],
            result['baseline']
        )
        
        # Преобразование в дБ для лучшей видимости
        epsilon = 1e-10
        spectrum_exp_db = 20 * np.log10(spectrum_exp + epsilon)
        spectrum_fit_db = 20 * np.log10(spectrum_fit + epsilon)
        
        # Построение
        ax = axes[i]
        ax.plot(freq, spectrum_exp_db, 'o', color='gray', markersize=3, 
               alpha=0.5, label='Эксперимент')
        ax.plot(freq, spectrum_fit_db, '-', color='red', linewidth=2, 
               label='Аппроксимация')
        
        # Маркеры пиков
        ax.axvline(x=result['f1'], color='blue', linestyle='--', 
                  linewidth=1, alpha=0.7, label=f"Пик 1: {result['f1']:.5f} ГГц")
        ax.axvline(x=result['f2'], color='red', linestyle='--', 
                  linewidth=1, alpha=0.7, label=f"Пик 2: {result['f2']:.5f} ГГц")
        
        # Настройка
        ax.set_xlabel('Частота (ГГц)', fontsize=11)
        ax.set_ylabel('|S| (дБ)', fontsize=11)
        ax.set_title(f'Поле: {field_value:.2f} Э, R²={result["r_squared"]:.6f}', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=9, loc='best')
    
    # Общий заголовок
    fig.suptitle('Примеры аппроксимации спектров двумя лоренцианами', 
                fontsize=14, fontweight='bold', y=0.995)
    
    # Плотная компоновка
    plt.tight_layout()
    
    # Сохранение графика
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  График сохранен: {save_path}")
    
    # Отображение на экране
    if show:
        plt.show()
    else:
        plt.close(fig)

