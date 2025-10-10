"""
Физические модели для антикроссинга резонатора и магнонов

Содержит теоретические модели S-параметров, расчеты частот и связей
"""

import numpy as np
import config_physics

# =============================================================================
# МОДЕЛЬ S-ПАРАМЕТРОВ
# =============================================================================

def anticrossing_one_mode_model(freq, field, params, s_type='S21'):
    """
    Теоретическая модель S-параметра для системы резонатор-магнон (одна мода)
    
    Parameters:
    -----------
    freq : array-like
        Массив частот (ГГц)
    field : array-like
        Массив магнитных полей (Э)
    params : dict
        Словарь параметров модели:
        - wc: резонансная частота резонатора (ГГц)
        - kappa: внешние потери резонатора (ГГц)
        - beta: внутренние потери резонатора (ГГц)
        - J: когерентная связь (ГГц)
        - Gamma: диссипативная связь (ГГц)
        - gamma: внешние потери магнонов (ГГц)
        - alpha: внутренние потери магнонов (ГГц)
        - gamma_g: гиромагнитное отношение (ГГц/Э)
        - H0: поле калибровки моды (Э)
        - w0: частота калибровки моды (ГГц)
    s_type : str
        Тип S-параметра ('S21' или 'S12')
        
    Returns:
    --------
    s_param : array-like
        Комплексный массив S-параметров
    """
    # Извлечение параметров
    wc = params['wc']
    kappa = params['kappa']
    beta = params['beta']
    J = params['J']
    gamma = params['gamma']
    alpha = params['alpha']
    
    # Gamma может быть передана явно или вычислена из kappa и gamma
    if 'Gamma' in params:
        Gamma = params['Gamma']
    else:
        Gamma = calculate_dissipative_coupling(kappa, gamma)
    
    gamma_g = config_physics.GYROMAGNETIC_RATIO  # Гиромагнитное отношение по умолчанию
    if 'gamma_g' in params:
        gamma_g = params['gamma_g']
    H0 = params.get('H0', 2900)  # Поле калибровки
    w0 = params.get('w0', 3.4)   # Частота калибровки
    
    omega_m = calculate_magnon_frequency_array(field, {'H0': H0, 'w0': w0, 'gamma_g': gamma_g})  # Частота магнонов в калибровке
    
    # Преобразование в радианные частоты (×2π для всех параметров кроме частот)
    omega = convert_to_radians(freq)
    omega_c = convert_to_radians(wc)
    kappa_rad = convert_to_radians(kappa)
    beta_rad = convert_to_radians(beta)
    J_rad = convert_to_radians(J)
    Gamma_rad = convert_to_radians(Gamma)
    gamma_rad = convert_to_radians(gamma)
    alpha_rad = convert_to_radians(alpha)
    omega_m = convert_to_radians(omega_m)
    
    # Знаменатель резонатора: i(ω - ωc) - (κ + β)
    cavity_denom = 1j * (omega - omega_c) - (kappa_rad + beta_rad)

    # Знаменатель магнонов: i(ω - ωm) - (γ + α)
    magnon_denom = 1j * (omega - omega_m) - (gamma_rad + alpha_rad)

    # Член связи: (iJ + Γ)²
    coupling_term = (1j * J_rad + Gamma_rad)**2
    
    # S-параметр: S = 1 + κ / (cavity_denom - coupling_term / magnon_denom)
    s_param = 1.0 + kappa_rad / (cavity_denom - coupling_term / magnon_denom)
    
    return s_param


def cavity_only_model(freq, wc, kappa, beta):
    """
    Модель S-параметра только для резонатора (без магнонов)
    
    Parameters:
    -----------
    freq : array-like
        Массив частот (ГГц)
    wc : float
        Резонансная частота резонатора (ГГц)
    kappa : float
        Внешние потери резонатора (ГГц)
    beta : float
        Внутренние потери резонатора (ГГц)
        
    Returns:
    --------
    s_cavity : array-like
        Комплексный массив S-параметров резонатора
    """
    # Преобразование в радианные частоты
    omega = convert_to_radians(freq)
    omega_c = convert_to_radians(wc)
    kappa_rad = convert_to_radians(kappa)
    beta_rad = convert_to_radians(beta)
    
    # Знаменатель резонатора: i(ω - ωc) - (κ + β)
    # (Та же формула, что в anticrossing_one_mode_model)
    cavity_denom = 1j * (omega - omega_c) - (kappa_rad + beta_rad)

    # S-параметр резонатора: S = 1 + κ / cavity_denom
    s_cavity = 1.0 + kappa_rad / cavity_denom
    
    return s_cavity


# =============================================================================
# РАСЧЕТ ЧАСТОТЫ МАГНОНОВ
# =============================================================================

def calculate_magnon_frequency_array(field, mode_params):
    """
    Рассчитать частоту магнонной моды для массива полей
    
    Parameters:
    -----------
    field : array-like
        Массив магнитных полей (Э)
    mode_params : dict
        Параметры магнонной моды:
        - H0: поле калибровки (Э)
        - w0: частота калибровки (ГГц)
        - gamma_g: гиромагнитное отношение (ГГц/Э), по умолчанию 2.8e-3
        
    Returns:
    --------
    freq : array-like
        Массив частот магнонной моды (ГГц)
        
    Формула: ωm = w0 + γ_g * (H - H0)
    """
    H0 = mode_params['H0']
    w0 = mode_params['w0']
    gamma_g = config_physics.GYROMAGNETIC_RATIO  # Гиромагнитное отношение по умолчанию
    if 'gamma_g' in mode_params:
        gamma_g = mode_params['gamma_g']
    
    # Линейная дисперсия магнонов
    freq = w0 + gamma_g * (field - H0)
    
    return freq


# =============================================================================
# РАСЧЕТ СВЯЗЕЙ
# =============================================================================

def calculate_dissipative_coupling(kappa, gamma):
    """
    Вычисление диссипативной связи из внешних потерь
    
    Parameters:
    -----------
    kappa : float
        Внешние потери резонатора (ГГц)
    gamma : float
        Внешние потери магнонной моды (ГГц)
    
    Returns:
    --------
    Gamma : float
        Коэффициент диссипативной связи (ГГц)
        
    Формула: Γ = √(κ * γ)
    """
    return np.sqrt(kappa * gamma)


# =============================================================================
# МУЛЬТИ-МОДОВАЯ ПОДДЕРЖКА
# =============================================================================

def anticrossing_multimode_model(freq, field, params, num_modes, s_type='S21'):
    """
    Теоретическая модель S-параметра для системы с несколькими магнонными модами
    
    Parameters:
    -----------
    freq : array-like
        Массив частот (ГГц)
    field : array-like
        Массив магнитных полей (Э)
    params : dict
        Словарь параметров модели:
        - wc, kappa, beta: параметры резонатора
        - J, Gamma, gamma, alpha: параметры связи и потерь
        - gamma_g: гиромагнитное отношение (ГГц/Э)
        - mode_calibrations: список [(H0_1, w0_1), (H0_2, w0_2), ...] для каждой моды
    num_modes : int
        Количество магнонных мод
    s_type : str
        Тип S-параметра ('S21' или 'S12')
        
    Returns:
    --------
    s_param : array-like
        Комплексный массив S-параметров
    """
    # Извлечение параметров резонатора и связи
    wc = params['wc']
    kappa = params['kappa']
    beta = params['beta']
    J = params['J']
    gamma = params['gamma']
    Gamma = calculate_dissipative_coupling(kappa, gamma)
    if 'Gamma' in params:
        Gamma = params['Gamma']
    alpha = params['alpha']
    gamma_g = config_physics.GYROMAGNETIC_RATIO  # Гиромагнитное отношение по умолчанию
    if 'gamma_g' in params:
        gamma_g = params['gamma_g']
    mode_calibrations = params.get('mode_calibrations', [])
    
    # Преобразование в радианные частоты
    omega = convert_to_radians(freq)
    omega_c = convert_to_radians(wc)
    kappa_rad = convert_to_radians(kappa)
    beta_rad = convert_to_radians(beta)
    J_rad = convert_to_radians(J)
    Gamma_rad = convert_to_radians(Gamma)
    gamma_rad = convert_to_radians(gamma)
    alpha_rad = convert_to_radians(alpha)
    
    # Знаменатель резонатора
    cavity_denom = 1j * (omega - omega_c) - (kappa_rad + beta_rad)
    
    # Член связи
    coupling_term = (1j * J_rad + Gamma_rad)**2
    
    # Суммирование вкладов от всех мод
    sum_mode_contributions = 0.0
    
    for i, mode in enumerate(mode_calibrations):
        H0, w0 = mode_calibrations[i]
        
        # Частота i-й магнонной моды
        omega_m_i = calculate_magnon_frequency_array(field, {'H0': H0, 'w0': w0, 'gamma_g': gamma_g})
        omega_m_i = convert_to_radians(omega_m_i)
        
        # Знаменатель i-й моды
        magnon_denom_i = 1j * (omega - omega_m_i) - (gamma_rad + alpha_rad)
        
        # Добавляем вклад i-й моды
        sum_mode_contributions += coupling_term / magnon_denom_i
    
    # S-параметр с учетом всех мод
    s_param = 1.0 + kappa_rad / (cavity_denom - sum_mode_contributions)
    
    return s_param


# =============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =============================================================================

def convert_to_radians(freq_ghz):
    """
    Конвертировать частоту из ГГц в рад/с
    
    Parameters:
    -----------
    freq_ghz : float or array-like
        Частота в ГГц
        
    Returns:
    --------
    freq_rad : float or array-like
        Частота в рад/с
        
    Формула: ω [рад/с] = 2π × f [ГГц] × 10^9
    """
    return 2.0 * np.pi * freq_ghz * 1e9


def convert_dB_to_linear(s_dB):
    """
    Конвертировать S-параметр из dB в линейную шкалу
    
    Parameters:
    -----------
    s_dB : float or array-like
        S-параметр в dB
        
    Returns:
    --------
    s_linear : float or array-like
        S-параметр в линейной шкале
        
    Формула: |S| = 10^(dB/20)
    """
    return 10 ** (s_dB / 20.0)


def convert_linear_to_dB(s_linear):
    """
    Конвертировать S-параметр из линейной шкалы в dB
    
    Parameters:
    -----------
    s_linear : float or array-like
        S-параметр в линейной шкале
        
    Returns:
    --------
    s_dB : float or array-like
        S-параметр в dB
        
    Формула: dB = 20 * log10(|S|)
    """
    return 20.0 * np.log10(np.abs(s_linear))


def validate_physical_constraints(params):
    """
    Проверить физические ограничения на параметры
    
    Parameters:
    -----------
    params : dict
        Словарь параметров
        
    Returns:
    --------
    valid : bool
        True если все параметры удовлетворяют физическим ограничениям
        
    Проверяет:
    - Все потери и связи должны быть положительными
    - Частоты должны быть положительными
    - Отсутствие NaN и Inf значений
    """
    # Проверка на NaN и Inf
    for key, value in params.items():
        if isinstance(value, (int, float)):
            if np.isnan(value) or np.isinf(value):
                raise ValueError(f"Параметр {key} содержит недопустимое значение: {value}")
    
    # Проверка положительности ключевых параметров
    positive_params = ['kappa', 'beta', 'gamma', 'alpha', 'J', 'Gamma', 'wc']
    for param in positive_params:
        if param in params:
            if params[param] <= 0:
                raise ValueError(f"Параметр {param} должен быть положительным, получено: {params[param]}")
    
    return True
