"""
Модуль отслеживания собственных частот системы (пиков в спектрах)

Вместо фитинга теоретической модели связанных осцилляторов,
этот модуль отслеживает реальные пики (максимумы или минимумы) в экспериментальных спектрах
и аппроксимирует их лоренцианами для извлечения собственных частот и ширин.

Автор: Alexey Kaminskiy
Дата создания: 2025-10-28
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, argrelmin
import config_physics


# =============================================================================
# МОДЕЛИ ЛОРЕНЦИАНА
# =============================================================================

def lorentzian_dip(freq, f0, width, amplitude, baseline, peak_type=None):
    """
    Универсальная модель лоренциана для пика в S-параметре
    
    Parameters:
    -----------
    freq : array-like
        Массив частот (ГГц)
    f0 : float
        Центральная частота пика (ГГц)
    width : float
        Ширина на полувысоте - FWHM (ГГц)
    amplitude : float
        Амплитуда пика (высота над/под базовой линией)
    baseline : float
        Базовая линия (уровень фона)
    peak_type : str or None
        Тип пика: 'maximum' для максимумов, 'minimum' для минимумов (провалов)
        Если None, берется из config_physics.PEAK_TYPE
        
    Returns:
    --------
    signal : array-like
        Значения сигнала
        
    Формула для maximum: S(f) = baseline + amplitude / (1 + ((f - f0) / (width/2))^2)
    Формула для minimum: S(f) = baseline - amplitude / (1 + ((f - f0) / (width/2))^2)
    """
    if peak_type is None:
        peak_type = config_physics.PEAK_TYPE
    
    lorentz = amplitude / (1 + ((freq - f0) / (width / 2.0))**2)
    if peak_type == 'maximum':
        return baseline + lorentz
    else:  # minimum
        return baseline - lorentz


def two_lorentzians_dip(freq, f1, w1, a1, f2, w2, a2, baseline, peak_type=None):
    """
    Модель двух лоренцианов (два пика - максимума или минимума)
    
    Parameters:
    -----------
    freq : array-like
        Массив частот (ГГц)
    f1, f2 : float
        Центральные частоты пиков (ГГц)
    w1, w2 : float
        Ширины на полувысоте - FWHM (ГГц)
    a1, a2 : float
        Амплитуды пиков
    baseline : float
        Базовая линия
    peak_type : str or None
        Тип пика: 'maximum' или 'minimum'. Если None, берется из config_physics.PEAK_TYPE
        
    Returns:
    --------
    signal : array-like
        Суммарный сигнал от двух лоренцианов
    """
    if peak_type is None:
        peak_type = config_physics.PEAK_TYPE
    
    L1 = lorentzian_dip(freq, f1, w1, a1, 0, peak_type)
    L2 = lorentzian_dip(freq, f2, w2, a2, 0, peak_type)
    return L1 + L2 + baseline


def fit_single_lorentzian(freq, spectrum, initial_f0, initial_width, peak_type=None, verbose=False):
    """
    Аппроксимировать спектр одним лоренцианом
    
    Parameters:
    -----------
    freq : array-like
        Массив частот (ГГц)
    spectrum : array-like
        Массив амплитуд S-параметра
    initial_f0 : float
        Начальная частота пика (ГГц)
    initial_width : float
        Начальная ширина (ГГц)
    peak_type : str or None
        Тип пика: 'maximum' или 'minimum'. Если None, берется из config_physics.PEAK_TYPE
    verbose : bool
        Выводить подробности фиттинга
        
    Returns:
    --------
    fitted_params : dict
        Подогнанные параметры {'f0', 'width', 'amplitude', 'baseline'}
    fit_quality : dict
        Метрики качества (R², RMSE)
    fitted_spectrum : array-like
        Восстановленный спектр
    """
    if peak_type is None:
        peak_type = config_physics.PEAK_TYPE
    
    # ФИКСИРОВАННЫЙ baseline - вычисляется по краям спектра и НЕ оптимизируется
    # Берем по 10% точек с каждого края для более стабильной оценки
    n_edge = max(5, len(spectrum) // 10)
    baseline_fixed = (np.mean(spectrum[:n_edge]) + np.mean(spectrum[-n_edge:])) / 2.0
    
    peak_idx = np.argmin(np.abs(freq - initial_f0))
    
    if peak_type == 'maximum':
        amplitude_est = spectrum[peak_idx] - baseline_fixed  # Для максимума: peak - baseline
    else:  # minimum
        amplitude_est = baseline_fixed - spectrum[peak_idx]  # Для минимума: baseline - peak
    
    # Начальные параметры БЕЗ baseline (он зафиксирован)
    p0 = [initial_f0, initial_width, amplitude_est]
    
    # Установка границ (БЕЗ baseline)
    freq_range = freq.max() - freq.min()
    spectrum_range = spectrum.max() - spectrum.min()
    
    bounds_lower = [
        freq.min() - freq_range * 0.1,
        0.001,
        0.0
    ]
    
    bounds_upper = [
        freq.max() + freq_range * 0.1,
        freq_range / 2.0,
        spectrum_range * 2.0
    ]
    
    try:
        # Создаем функцию-обертку с фиксированным peak_type и baseline
        def model_func(freq, f0, width, amplitude):
            return lorentzian_dip(freq, f0, width, amplitude, baseline_fixed, peak_type)
        
        popt, pcov = curve_fit(
            model_func,
            freq,
            spectrum,
            p0=p0,
            bounds=(bounds_lower, bounds_upper),
            method='trf',
            maxfev=10000,
            loss='soft_l1'
        )
        
        fitted_params = {
            'f0': popt[0],
            'width': popt[1],
            'amplitude': popt[2],
            'baseline': baseline_fixed  # Фиксированное значение
        }
        
        fitted_spectrum = lorentzian_dip(freq, popt[0], popt[1], popt[2], baseline_fixed, peak_type)
        
        # Качество подгонки
        residuals = spectrum - fitted_spectrum
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((spectrum - np.mean(spectrum))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        rmse = np.sqrt(np.mean(residuals**2))
        
        fit_quality = {
            'r_squared': r_squared,
            'rmse': rmse,
            'residuals': residuals
        }
        
        if verbose:
            print(f"  ✓ Одиночный фит: R²={r_squared:.6f}, f={fitted_params['f0']:.6f} ГГц, w={fitted_params['width']:.6f} ГГц")
        
        return fitted_params, fit_quality, fitted_spectrum
        
    except Exception as e:
        if verbose:
            print(f"  ✗ Ошибка фиттинга одиночного пика: {e}")
        return None, None, None


# =============================================================================
# ПОИСК ПИКОВ
# =============================================================================

def find_nearest_peak(data, freq_click, field_click, search_radius_freq=0.05, peak_type=None):
    """
    Найти ближайший локальный пик (максимум или минимум) к точке клика
    
    Ищет локальные экстремумы ТОЛЬКО на спектре для указанного поля.
    Используется для уточнения положения пика после клика пользователя.
    
    Parameters:
    -----------
    data : dict
        Данные S-параметров:
        - 'freq': массив частот (ГГц)
        - 'field': массив полей (Э)
        - 's_param': 2D массив S-параметров
    freq_click : float
        Частота клика (ГГц)
    field_click : float
        Поле клика (Э) - будет использовано ближайшее доступное значение поля
    search_radius_freq : float
        Радиус поиска по частоте (ГГц), по умолчанию ±0.05 ГГц
    peak_type : str or None
        Тип пика: 'maximum' или 'minimum'. Если None, берется из config_physics.PEAK_TYPE
        
    Returns:
    --------
    peak_location : tuple
        Местоположение пика (field, freq)
    peak_value : float
        Значение амплитуды в пике
    """
    if peak_type is None:
        peak_type = config_physics.PEAK_TYPE
    
    freq = data['freq']
    field = data['field']
    s_param = data['s_param']
    
    # Найти ближайшее значение поля к клику
    field_idx = np.argmin(np.abs(field - field_click))
    field_actual = field[field_idx]
    
    print(f"  Клик: H={field_click:.2f} Э → используется ближайшее поле H={field_actual:.2f} Э")
    
    # Извлечь спектр для этого поля
    spectrum = np.abs(s_param[field_idx, :])
    
    # Определить диапазон поиска по частоте
    freq_min = freq_click - search_radius_freq
    freq_max = freq_click + search_radius_freq
    
    # Маска для фильтрации частот
    freq_mask = (freq >= freq_min) & (freq <= freq_max)
    
    # Проверка наличия данных
    if not np.any(freq_mask):
        print(f"  ⚠ Нет данных в окрестности клика (f={freq_click:.3f} ГГц)")
        return (field_actual, freq_click), None
    
    # Извлечь участок спектра
    freq_subset = freq[freq_mask]
    spectrum_subset = spectrum[freq_mask]
    
    # Найти экстремум в области
    if peak_type == 'maximum':
        extremum_idx = np.argmax(spectrum_subset)
        peak_type_ru = "максимум"
    else:  # minimum
        extremum_idx = np.argmin(spectrum_subset)
        peak_type_ru = "минимум"
    
    freq_peak = freq_subset[extremum_idx]
    peak_value = spectrum_subset[extremum_idx]
    
    print(f"  ✓ Найден {peak_type_ru}: H={field_actual:.2f} Э, f={freq_peak:.6f} ГГц, |S|={peak_value:.6f}")
    
    return (field_actual, freq_peak), peak_value


def find_local_minima_in_spectrum(freq, spectrum, prominence=0.01, distance=10, peak_type=None):
    """
    Найти все локальные экстремумы (максимумы или минимумы) в спектре
    
    Parameters:
    -----------
    freq : array-like
        Массив частот
    spectrum : array-like
        Массив амплитуд S-параметра
    prominence : float
        Минимальная выраженность пика
    distance : int
        Минимальное расстояние между пиками (в точках)
    peak_type : str or None
        Тип пика: 'maximum' или 'minimum'. Если None, берется из config_physics.PEAK_TYPE
        
    Returns:
    --------
    peak_indices : array-like
        Индексы локальных экстремумов
    peak_properties : dict
        Свойства пиков (prominences, widths, etc.)
    """
    if peak_type is None:
        peak_type = config_physics.PEAK_TYPE
    
    # Поиск пиков
    if peak_type == 'maximum':
        # Ищем максимумы
        peak_indices, properties = find_peaks(
            spectrum,
            prominence=prominence,
            distance=distance
        )
    else:  # minimum
        # Ищем минимумы (инвертируем спектр)
        peak_indices, properties = find_peaks(
            -spectrum,
            prominence=prominence,
            distance=distance
        )
    
    return peak_indices, properties


# =============================================================================
# ОЦЕНКА ПАРАМЕТРОВ ПИКОВ
# =============================================================================

def estimate_peak_width(freq, spectrum, peak_idx):
    """
    Оценить ширину пика на полувысоте (FWHM)
    
    Parameters:
    -----------
    freq : array-like
        Массив частот
    spectrum : array-like
        Массив амплитуд S-параметра
    peak_idx : int
        Индекс пика (минимума) в массиве
        
    Returns:
    --------
    width : float
        Ширина пика на полувысоте (FWHM)
    """
    # Значение в минимуме
    min_value = spectrum[peak_idx]
    
    # Оценка базовой линии (среднее по краям спектра)
    baseline = (np.mean(spectrum[:5]) + np.mean(spectrum[-5:])) / 2.0
    
    # Полувысота: (baseline + min_value) / 2
    half_height = (baseline + min_value) / 2.0
    
    # Поиск точек пересечения полувысоты слева и справа от минимума
    # Слева от пика
    left_indices = np.where(spectrum[:peak_idx] <= half_height)[0]
    if len(left_indices) > 0:
        left_idx = left_indices[0]  # Первый индекс слева, где спектр ниже полувысоты
    else:
        left_idx = 0
    
    # Справа от пика
    right_indices = np.where(spectrum[peak_idx:] <= half_height)[0]
    if len(right_indices) > 0:
        right_idx = peak_idx + right_indices[-1]  # Последний индекс справа
    else:
        right_idx = len(spectrum) - 1
    
    # Ширина
    width = np.abs(freq[right_idx] - freq[left_idx])
    
    # Минимальная ширина (для избежания нулевых значений)
    if width < 0.001:  # Менее 1 МГц
        width = 0.005  # Установить минимум 5 МГц
    
    return width


def estimate_peak_parameters(freq, spectrum, peak_freq_guess):
    """
    Оценить параметры пика (частота, ширина, амплитуда) для начального приближения
    
    Parameters:
    -----------
    freq : array-like
        Массив частот
    spectrum : array-like
        Массив амплитуд S-параметра
    peak_freq_guess : float
        Примерная частота пика
        
    Returns:
    --------
    peak_params : dict
        Параметры пика:
        - 'f0': частота (ГГц)
        - 'width': ширина FWHM (ГГц)
        - 'amplitude': амплитуда провала
        - 'baseline': базовая линия
    """
    # Найти индекс ближайшей частоты
    peak_idx = np.argmin(np.abs(freq - peak_freq_guess))
    
    # Уточнить минимум в окрестности (±10 точек)
    search_window = 10
    start_idx = max(0, peak_idx - search_window)
    end_idx = min(len(spectrum), peak_idx + search_window + 1)
    local_spectrum = spectrum[start_idx:end_idx]
    local_min_idx = np.argmin(local_spectrum)
    peak_idx = start_idx + local_min_idx
    
    # Частота пика
    f0 = freq[peak_idx]
    
    # Значение в минимуме
    min_value = spectrum[peak_idx]
    
    # Оценка базовой линии
    baseline = (np.mean(spectrum[:5]) + np.mean(spectrum[-5:])) / 2.0
    
    # Амплитуда провала
    amplitude = baseline - min_value
    
    # Ширина
    width = estimate_peak_width(freq, spectrum, peak_idx)
    
    peak_params = {
        'f0': f0,
        'width': width,
        'amplitude': amplitude,
        'baseline': baseline
    }
    
    return peak_params


# =============================================================================
# ФИТТИНГ ЛОРЕНЦИАНАМИ
# =============================================================================

def fit_spectrum_with_two_lorentzians(freq, spectrum, initial_params, verbose=False):
    """
    Аппроксимировать спектр двумя лоренцианами
    
    Parameters:
    -----------
    freq : array-like
        Массив частот (ГГц)
    spectrum : array-like
        Массив амплитуд S-параметра
    initial_params : dict
        Начальные параметры:
        - 'f1', 'w1', 'a1': параметры первого пика
        - 'f2', 'w2', 'a2': параметры второго пика
        - 'baseline': базовая линия
    verbose : bool
        Выводить подробности фиттинга
        
    Returns:
    --------
    fitted_params : dict
        Подогнанные параметры пиков
    fit_quality : dict
        Метрики качества подгонки (R², RMSE)
    fitted_spectrum : array-like
        Восстановленный спектр
    """
    # Извлечение начальных параметров
    p0 = [
        initial_params['f1'],
        initial_params['w1'],
        initial_params['a1'],
        initial_params['f2'],
        initial_params['w2'],
        initial_params['a2'],
        initial_params['baseline']
    ]
    
    # Установка границ параметров
    # Частоты: в пределах диапазона данных ± небольшой запас
    freq_min, freq_max = freq.min(), freq.max()
    freq_range = freq_max - freq_min
    
    # Ширины: от 0.001 до половины диапазона частот
    width_min = 0.001
    width_max = freq_range / 2.0
    
    # Амплитуды: от 0 до максимального размаха спектра
    spectrum_range = np.max(spectrum) - np.min(spectrum)
    amp_min = 0.0
    amp_max = spectrum_range * 2.0
    
    # Базовая линия: в пределах диапазона значений спектра ± запас
    baseline_min = np.min(spectrum) - spectrum_range * 0.5
    baseline_max = np.max(spectrum) + spectrum_range * 0.5
    
    bounds_lower = [
        freq_min - freq_range * 0.1, width_min, amp_min,  # Пик 1
        freq_min - freq_range * 0.1, width_min, amp_min,  # Пик 2
        baseline_min
    ]
    
    bounds_upper = [
        freq_max + freq_range * 0.1, width_max, amp_max,  # Пик 1
        freq_max + freq_range * 0.1, width_max, amp_max,  # Пик 2
        baseline_max
    ]
    
    try:
        # Выполнение фиттинга
        popt, pcov = curve_fit(
            two_lorentzians_dip,
            freq,
            spectrum,
            p0=p0,
            bounds=(bounds_lower, bounds_upper),
            method='trf',
            maxfev=10000,
            ftol=1e-10,
            xtol=1e-10,
            loss='soft_l1'  # Робастная функция потерь
        )
        
        # Извлечение результатов
        fitted_params = {
            'f1': popt[0],
            'w1': popt[1],
            'a1': popt[2],
            'f2': popt[3],
            'w2': popt[4],
            'a2': popt[5],
            'baseline': popt[6]
        }
        
        # Восстановленный спектр
        fitted_spectrum = two_lorentzians_dip(freq, *popt)
        
        # Вычисление качества подгонки
        residuals = spectrum - fitted_spectrum
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((spectrum - np.mean(spectrum))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        rmse = np.sqrt(np.mean(residuals**2))
        
        fit_quality = {
            'r_squared': r_squared,
            'rmse': rmse,
            'residuals': residuals
        }
        
        if verbose:
            print(f"  ✓ Фит успешен: R²={r_squared:.6f}, RMSE={rmse:.6f}")
            print(f"    Пик 1: f={fitted_params['f1']:.6f} ГГц, w={fitted_params['w1']:.6f} ГГц")
            print(f"    Пик 2: f={fitted_params['f2']:.6f} ГГц, w={fitted_params['w2']:.6f} ГГц")
        
        return fitted_params, fit_quality, fitted_spectrum
        
    except Exception as e:
        if verbose:
            print(f"  ✗ Ошибка фиттинга: {e}")
        return None, None, None


# =============================================================================
# РАЗДЕЛЬНЫЙ ФИТТИНГ ДВУХ ПИКОВ
# =============================================================================

def fit_two_peaks_separately(freq, spectrum, range1, range2, peak_type=None, verbose=False):
    """
    Фиттинг двух пиков РАЗДЕЛЬНО в своих диапазонах
    
    Каждый пик фитится независимо своим лоренцианом.
    Возвращает параметры обоих пиков и восстановленный спектр как сумму двух лоренцианов.
    
    Parameters:
    -----------
    freq : array-like
        Массив частот (ГГц)
    spectrum : array-like
        Массив амплитуд S-параметра
    range1 : tuple
        Диапазон первого пика (freq_min, freq_max) в ГГц
    range2 : tuple
        Диапазон второго пика (freq_min, freq_max) в ГГц
    peak_type : str or None
        Тип пика: 'maximum' или 'minimum'. Если None, берется из config_physics.PEAK_TYPE
    verbose : bool
        Выводить подробности
        
    Returns:
    --------
    fitted_params : dict
        Подогнанные параметры {'f1', 'w1', 'a1', 'baseline1', 'f2', 'w2', 'a2', 'baseline2'}
    fit_quality : dict
        Качество подгонки (R², RMSE) для каждого пика отдельно
    fitted_spectra : dict
        Восстановленные спектры: {'peak1', 'peak2', 'combined'}
    ranges : tuple
        Использованные диапазоны (range1, range2)
    """
    if peak_type is None:
        peak_type = config_physics.PEAK_TYPE
    
    # Маски для диапазонов
    mask1 = (freq >= range1[0]) & (freq <= range1[1])
    mask2 = (freq >= range2[0]) & (freq <= range2[1])
    
    if not np.any(mask1) or not np.any(mask2):
        if verbose:
            print(f"  ✗ Пустые диапазоны для фиттинга")
        return None, None, None, (range1, range2)
    
    # Извлечь данные из диапазонов
    freq1 = freq[mask1]
    spectrum1 = spectrum[mask1]
    freq2 = freq[mask2]
    spectrum2 = spectrum[mask2]
    
    if verbose:
        print(f"  Раздельный фиттинг двух пиков:")
        print(f"    Диапазон 1: [{range1[0]:.6f}, {range1[1]:.6f}] ГГц ({len(freq1)} точек)")
        print(f"    Диапазон 2: [{range2[0]:.6f}, {range2[1]:.6f}] ГГц ({len(freq2)} точек)")
    
    # Найти экстремумы в диапазонах
    if peak_type == 'maximum':
        extremum_idx1 = np.argmax(spectrum1)
        extremum_idx2 = np.argmax(spectrum2)
    else:  # minimum
        extremum_idx1 = np.argmin(spectrum1)
        extremum_idx2 = np.argmin(spectrum2)
    
    f1_init = freq1[extremum_idx1]
    f2_init = freq2[extremum_idx2]
    
    # Начальные ширины
    w1_init = (range1[1] - range1[0]) / 3.0
    w2_init = (range2[1] - range2[0]) / 3.0
    
    # Фит первого пика
    try:
        params1, quality1, fitted1 = fit_single_lorentzian(
            freq1, spectrum1, f1_init, w1_init, peak_type=peak_type, verbose=False
        )
        if params1 is None:
            if verbose:
                print(f"  ✗ Ошибка фиттинга пика 1")
            return None, None, None, (range1, range2)
    except Exception as e:
        if verbose:
            print(f"  ✗ Ошибка фиттинга пика 1: {e}")
        return None, None, None, (range1, range2)
    
    # Фит второго пика
    try:
        params2, quality2, fitted2 = fit_single_lorentzian(
            freq2, spectrum2, f2_init, w2_init, peak_type=peak_type, verbose=False
        )
        if params2 is None:
            if verbose:
                print(f"  ✗ Ошибка фиттинга пика 2")
            return None, None, None, (range1, range2)
    except Exception as e:
        if verbose:
            print(f"  ✗ Ошибка фиттинга пика 2: {e}")
        return None, None, None, (range1, range2)
    
    # Объединить параметры
    fitted_params = {
        'f1': params1['f0'],
        'w1': params1['width'],
        'a1': params1['amplitude'],
        'baseline1': params1['baseline'],
        'f2': params2['f0'],
        'w2': params2['width'],
        'a2': params2['amplitude'],
        'baseline2': params2['baseline'],
        'baseline': (params1['baseline'] + params2['baseline']) / 2.0  # Усредненная базовая линия
    }
    
    # Восстановить спектры для ПОЛНОГО диапазона частот
    fitted_spectrum1_full = lorentzian_dip(freq, params1['f0'], params1['width'], 
                                           params1['amplitude'], params1['baseline'], peak_type)
    fitted_spectrum2_full = lorentzian_dip(freq, params2['f0'], params2['width'], 
                                           params2['amplitude'], params2['baseline'], peak_type)
    
    # Комбинированный спектр (сумма двух лоренцианов минус дублированная базовая линия)
    fitted_spectrum_combined = fitted_spectrum1_full + fitted_spectrum2_full - fitted_params['baseline']
    
    fitted_spectra = {
        'peak1': fitted_spectrum1_full,
        'peak2': fitted_spectrum2_full,
        'combined': fitted_spectrum_combined
    }
    
    # Качество подгонки для каждого пика в его диапазоне
    fit_quality = {
        'r_squared_peak1': quality1['r_squared'],
        'rmse_peak1': quality1['rmse'],
        'r_squared_peak2': quality2['r_squared'],
        'rmse_peak2': quality2['rmse'],
        'r_squared': (quality1['r_squared'] + quality2['r_squared']) / 2.0,  # Усредненное качество
        'rmse': np.sqrt((quality1['rmse']**2 + quality2['rmse']**2) / 2.0),  # Усредненная ошибка
        'residuals': spectrum - fitted_spectrum_combined  # Для полного спектра
    }
    
    if verbose:
        print(f"  ✓ Раздельный фит успешен:")
        print(f"    Пик 1: f={fitted_params['f1']:.6f} ГГц, w={fitted_params['w1']:.6f} ГГц, R²={quality1['r_squared']:.6f}")
        print(f"    Пик 2: f={fitted_params['f2']:.6f} ГГц, w={fitted_params['w2']:.6f} ГГц, R²={quality2['r_squared']:.6f}")
        print(f"    Среднее: R²={fit_quality['r_squared']:.6f}, RMSE={fit_quality['rmse']:.6f}")
    
    return fitted_params, fit_quality, fitted_spectra, (range1, range2)


def fit_two_lorentzians_in_ranges(freq, spectrum, range1, range2, peak_type=None, fit_mode=None, verbose=False):
    """
    Фиттинг двух лоренцианов в заданных диапазонах частот (без внутренней итерации)
    
    Режим фиттинга определяется параметром fit_mode:
    - 'combined': оба пика фитятся одновременно (может быть нестабильно для слабо перекрывающихся пиков)
    - 'separate': каждый пик фитится отдельно в своем диапазоне (более стабильно)
    
    ВНИМАНИЕ: Эта функция выполняет только ОДИН фиттинг в заданных диапазонах.
    Проверка и корректировка диапазонов должна выполняться вызывающей функцией.
    
    Parameters:
    -----------
    freq : array-like
        Массив частот (ГГц)
    spectrum : array-like
        Массив амплитуд S-параметра
    range1 : tuple
        Диапазон первого пика (freq_min, freq_max) в ГГц
    range2 : tuple
        Диапазон второго пика (freq_min, freq_max) в ГГц
    peak_type : str or None
        Тип пика: 'maximum' или 'minimum'. Если None, берется из config_physics.PEAK_TYPE
    fit_mode : str or None
        Режим фиттинга: 'combined' или 'separate'. Если None, берется из config_physics.FIT_MODE
    verbose : bool
        Выводить подробности
        
    Returns:
    --------
    fitted_params : dict
        Подогнанные параметры {'f1', 'w1', 'a1', 'f2', 'w2', 'a2', 'baseline', ...}
    fit_quality : dict
        Качество подгонки (R², RMSE)
    fitted_spectrum : array-like or dict
        Для 'combined': array восстановленного спектра
        Для 'separate': dict с ключами {'peak1', 'peak2', 'combined'}
    ranges : tuple
        Использованные диапазоны (range1, range2)
    """
    if peak_type is None:
        peak_type = config_physics.PEAK_TYPE
    
    if fit_mode is None:
        fit_mode = config_physics.FIT_MODE
    
    # Режим раздельного фиттинга
    if fit_mode == 'separate':
        return fit_two_peaks_separately(freq, spectrum, range1, range2, peak_type=peak_type, verbose=verbose)
    
    # Режим комбинированного фиттинга
    # Маски для диапазонов
    mask1 = (freq >= range1[0]) & (freq <= range1[1])
    mask2 = (freq >= range2[0]) & (freq <= range2[1])
    
    if not np.any(mask1) or not np.any(mask2):
        if verbose:
            print(f"  ✗ Пустые диапазоны для фиттинга")
        return None, None, None, (range1, range2)
    
    # Извлечь данные из диапазонов
    freq1 = freq[mask1]
    spectrum1 = spectrum[mask1]
    freq2 = freq[mask2]
    spectrum2 = spectrum[mask2]
    
    # Найти экстремумы в диапазонах (центры пиков)
    if peak_type == 'maximum':
        extremum_idx1 = np.argmax(spectrum1)
        extremum_idx2 = np.argmax(spectrum2)
    else:  # minimum
        extremum_idx1 = np.argmin(spectrum1)
        extremum_idx2 = np.argmin(spectrum2)
    
    f1_init = freq1[extremum_idx1]
    f2_init = freq2[extremum_idx2]
    
    # Ширины = половина диапазона
    w1_init = (range1[1] - range1[0]) / 2.0
    w2_init = (range2[1] - range2[0]) / 2.0
    
    # Оценка амплитуд и базовой линии
    baseline_est = (np.mean(spectrum[:10]) + np.mean(spectrum[-10:])) / 2.0
    
    if peak_type == 'maximum':
        a1_init = spectrum1[extremum_idx1] - baseline_est
        a2_init = spectrum2[extremum_idx2] - baseline_est
    else:  # minimum
        a1_init = baseline_est - spectrum1[extremum_idx1]
        a2_init = baseline_est - spectrum2[extremum_idx2]
    
    if verbose:
        print(f"  Начальные параметры:")
        print(f"    Пик 1: f={f1_init:.6f} ГГц, w={w1_init:.6f} ГГц, диапазон=[{range1[0]:.6f}, {range1[1]:.6f}]")
        print(f"    Пик 2: f={f2_init:.6f} ГГц, w={w2_init:.6f} ГГц, диапазон=[{range2[0]:.6f}, {range2[1]:.6f}]")
    
    # Объединить данные из обоих диапазонов для фиттинга
    freq_fit = np.concatenate([freq1, freq2])
    spectrum_fit = np.concatenate([spectrum1, spectrum2])
    
    # ФИКСИРОВАННЫЙ baseline - вычисляется по краям ПОЛНОГО спектра
    n_edge = max(10, len(spectrum) // 10)
    baseline_fixed = (np.mean(spectrum[:n_edge]) + np.mean(spectrum[-n_edge:])) / 2.0
    
    # Начальные параметры БЕЗ baseline (он зафиксирован)
    p0 = [f1_init, w1_init, a1_init, f2_init, w2_init, a2_init]
    
    # Границы параметров (БЕЗ baseline)
    spectrum_range = spectrum.max() - spectrum.min()
    
    bounds_lower = [
        range1[0], 0.001, 0.0,  # Пик 1: частота в пределах диапазона
        range2[0], 0.001, 0.0   # Пик 2: частота в пределах диапазона
    ]
    
    bounds_upper = [
        range1[1], (range1[1] - range1[0]) * 2, spectrum_range * 2.0,  # Пик 1
        range2[1], (range2[1] - range2[0]) * 2, spectrum_range * 2.0   # Пик 2
    ]
    
    try:
        # Создаем функцию-обертку с фиксированным peak_type и baseline
        def model_func(freq, f1, w1, a1, f2, w2, a2):
            return two_lorentzians_dip(freq, f1, w1, a1, f2, w2, a2, baseline_fixed, peak_type)
        
        # Фиттинг
        popt, pcov = curve_fit(
            model_func,
            freq_fit,
            spectrum_fit,
            p0=p0,
            bounds=(bounds_lower, bounds_upper),
            method='trf',
            maxfev=10000,
            loss='soft_l1'
        )
        
        fitted_params = {
            'f1': popt[0],
            'w1': popt[1],
            'a1': popt[2],
            'f2': popt[3],
            'w2': popt[4],
            'a2': popt[5],
            'baseline': baseline_fixed  # Фиксированное значение
        }
        
        # Восстановить спектр для ПОЛНОГО диапазона частот
        fitted_spectrum_full = two_lorentzians_dip(
            freq, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], baseline_fixed, peak_type
        )
        
        # Качество подгонки ТОЛЬКО для диапазонов фиттинга
        fitted_in_ranges = two_lorentzians_dip(
            freq_fit, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], baseline_fixed, peak_type
        )
        residuals_in_ranges = spectrum_fit - fitted_in_ranges
        ss_res = np.sum(residuals_in_ranges**2)
        ss_tot = np.sum((spectrum_fit - np.mean(spectrum_fit))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        rmse = np.sqrt(np.mean(residuals_in_ranges**2))
        
        # Полные residuals для всего спектра (для визуализации)
        residuals_full = spectrum - fitted_spectrum_full
        
        fit_quality = {
            'r_squared': r_squared,
            'rmse': rmse,
            'residuals': residuals_full,  # Для полного спектра
            'r_squared_in_ranges': r_squared,  # Явно указываем, что это для диапазонов
            'rmse_in_ranges': rmse
        }
        
        if verbose:
            print(f"  ✓ Фит успешен: R²={r_squared:.6f}, RMSE={rmse:.6f} (в диапазонах)")
            print(f"    Пик 1: f={fitted_params['f1']:.6f} ГГц, w={fitted_params['w1']:.6f} ГГц")
            print(f"    Пик 2: f={fitted_params['f2']:.6f} ГГц, w={fitted_params['w2']:.6f} ГГц")
        
        return fitted_params, fit_quality, fitted_spectrum_full, (range1, range2)
        
    except Exception as e:
        if verbose:
            print(f"  ✗ Ошибка фиттинга: {e}")
        return None, None, None, (range1, range2)


def validate_and_adjust_ranges(fitted_params, current_ranges, centering_threshold=0.15, width_tolerance=0.3, min_range_width=0.005):
    """
    Проверить корректность диапазонов фиттинга и вычислить оптимальные диапазоны при необходимости
    
    Проверяет два критерия:
    1. Центрирование: пик должен быть близко к центру диапазона
    2. Соотношение ширин: ширина диапазона должна соответствовать ширине пика
    
    Parameters:
    -----------
    fitted_params : dict
        Подогнанные параметры {'f1', 'w1', 'f2', 'w2', ...}
    current_ranges : tuple
        Текущие диапазоны (range1, range2), где range = (freq_min, freq_max)
    centering_threshold : float
        Максимально допустимое смещение пика от центра диапазона (в долях ширины диапазона)
        По умолчанию 0.15 (15%)
    width_tolerance : float
        Допустимое отклонение соотношения ширин (в долях)
        По умолчанию 0.3 (30% от целевого соотношения)
    min_range_width : float
        Минимально допустимая ширина диапазона в ГГц (по умолчанию 0.005 = 5 МГц)
    
    Returns:
    --------
    status : str
        'ok' - диапазоны корректны, фиттинг можно принять
        'adjust' - диапазоны нужно скорректировать
        'too_narrow' - диапазоны слишком узкие, корректировка невозможна
    new_ranges : tuple or None
        Новые оптимальные диапазоны (range1, range2) или None если корректировка не нужна
    diagnostics : dict
        Подробная информация о проверке
    """
    range1, range2 = current_ranges
    
    # Параметры текущих диапазонов
    range1_center = (range1[0] + range1[1]) / 2.0
    range2_center = (range2[0] + range2[1]) / 2.0
    range1_width = range1[1] - range1[0]
    range2_width = range2[1] - range2[0]
    
    # Проверка на минимальную ширину диапазона
    if range1_width < min_range_width or range2_width < min_range_width:
        diagnostics = {
            'peak1': {'range_width': range1_width, 'too_narrow': range1_width < min_range_width},
            'peak2': {'range_width': range2_width, 'too_narrow': range2_width < min_range_width}
        }
        return 'too_narrow', None, diagnostics
    
    # Целевая ширина диапазона (по RANGE_WIDTH_MULTIPLIER из config)
    # Но не меньше минимальной ширины
    target_range1_width = max(config_physics.RANGE_WIDTH_MULTIPLIER * fitted_params['w1'], min_range_width)
    target_range2_width = max(config_physics.RANGE_WIDTH_MULTIPLIER * fitted_params['w2'], min_range_width)
    
    # Проверка 1: Центрирование пиков
    shift1 = abs(fitted_params['f1'] - range1_center)
    shift2 = abs(fitted_params['f2'] - range2_center)
    rel_shift1 = shift1 / range1_width if range1_width > 0 else 0
    rel_shift2 = shift2 / range2_width if range2_width > 0 else 0
    
    centering_issue1 = rel_shift1 > centering_threshold
    centering_issue2 = rel_shift2 > centering_threshold
    
    # Проверка 2: Соотношение ширин диапазона и пика
    width_ratio1 = range1_width / target_range1_width if target_range1_width > 0 else 1.0
    width_ratio2 = range2_width / target_range2_width if target_range2_width > 0 else 1.0
    
    # Допустимый диапазон соотношений: [1-tolerance, 1+tolerance]
    width_issue1 = abs(width_ratio1 - 1.0) > width_tolerance
    width_issue2 = abs(width_ratio2 - 1.0) > width_tolerance
    
    # Общее решение
    needs_adjustment1 = centering_issue1 or width_issue1
    needs_adjustment2 = centering_issue2 or width_issue2
    needs_adjustment = needs_adjustment1 or needs_adjustment2
    
    # Диагностика
    diagnostics = {
        'peak1': {
            'freq': fitted_params['f1'],
            'width': fitted_params['w1'],
            'range_center': range1_center,
            'range_width': range1_width,
            'target_range_width': target_range1_width,
            'shift': shift1,
            'rel_shift': rel_shift1,
            'width_ratio': width_ratio1,
            'centering_issue': centering_issue1,
            'width_issue': width_issue1,
            'needs_adjustment': needs_adjustment1
        },
        'peak2': {
            'freq': fitted_params['f2'],
            'width': fitted_params['w2'],
            'range_center': range2_center,
            'range_width': range2_width,
            'target_range_width': target_range2_width,
            'shift': shift2,
            'rel_shift': rel_shift2,
            'width_ratio': width_ratio2,
            'centering_issue': centering_issue2,
            'width_issue': width_issue2,
            'needs_adjustment': needs_adjustment2
        }
    }
    
    # Если корректировка не нужна - вернуть 'ok'
    if not needs_adjustment:
        return 'ok', None, diagnostics
    
    # Вычисление новых оптимальных диапазонов
    new_range1 = range1
    new_range2 = range2
    
    if needs_adjustment1:
        # Центрируем на частоте пика с правильной шириной
        half_width1 = target_range1_width / 2.0
        new_range1 = (fitted_params['f1'] - half_width1, fitted_params['f1'] + half_width1)
    
    if needs_adjustment2:
        half_width2 = target_range2_width / 2.0
        new_range2 = (fitted_params['f2'] - half_width2, fitted_params['f2'] + half_width2)
    
    new_ranges = (new_range1, new_range2)
    
    return 'adjust', new_ranges, diagnostics



def update_ranges_by_shift(freq, spectrum, prev_ranges, prev_peak_freqs, peak_type=None):
    """
    Обновить диапазоны фиттинга на основе смещения экстремумов
    
    Находит экстремумы в старых диапазонах, вычисляет их смещение
    относительно предыдущих положений и сдвигает диапазоны на эту величину.
    Ширина диапазонов не меняется.
    
    Parameters:
    -----------
    freq : array-like
        Массив частот текущего спектра (ГГц)
    spectrum : array-like
        Массив амплитуд текущего спектра
    prev_ranges : tuple
        Предыдущие диапазоны (range1, range2)
    prev_peak_freqs : tuple
        Частоты пиков в предыдущем спектре (f1, f2)
    peak_type : str or None
        Тип пика: 'maximum' или 'minimum'. Если None, берется из config_physics.PEAK_TYPE
        
    Returns:
    --------
    new_ranges : tuple
        Обновленные диапазоны (new_range1, new_range2)
    """
    if peak_type is None:
        peak_type = config_physics.PEAK_TYPE
    
    prev_range1, prev_range2 = prev_ranges
    prev_f1, prev_f2 = prev_peak_freqs
    
    # Ширины диапазонов (не меняются)
    width1 = prev_range1[1] - prev_range1[0]
    width2 = prev_range2[1] - prev_range2[0]
    
    # Найти экстремумы в предыдущих диапазонах
    mask1 = (freq >= prev_range1[0]) & (freq <= prev_range1[1])
    mask2 = (freq >= prev_range2[0]) & (freq <= prev_range2[1])
    
    # Если диапазоны вышли за пределы - расширяем поиск
    if not np.any(mask1):
        # Ищем вблизи предыдущей частоты
        search_margin = width1
        mask1 = (freq >= prev_f1 - search_margin) & (freq <= prev_f1 + search_margin)
    
    if not np.any(mask2):
        search_margin = width2
        mask2 = (freq >= prev_f2 - search_margin) & (freq <= prev_f2 + search_margin)
    
    # Найти новые положения экстремумов
    if np.any(mask1):
        spectrum1 = spectrum[mask1]
        freq1 = freq[mask1]
        if peak_type == 'maximum':
            extremum_idx1 = np.argmax(spectrum1)
        else:  # minimum
            extremum_idx1 = np.argmin(spectrum1)
        new_f1 = freq1[extremum_idx1]
        shift1 = new_f1 - prev_f1
    else:
        shift1 = 0.0
        new_f1 = prev_f1
    
    if np.any(mask2):
        spectrum2 = spectrum[mask2]
        freq2 = freq[mask2]
        if peak_type == 'maximum':
            extremum_idx2 = np.argmax(spectrum2)
        else:  # minimum
            extremum_idx2 = np.argmin(spectrum2)
        new_f2 = freq2[extremum_idx2]
        shift2 = new_f2 - prev_f2
    else:
        shift2 = 0.0
        new_f2 = prev_f2
    
    # Сдвинуть диапазоны
    new_range1 = (prev_range1[0] + shift1, prev_range1[1] + shift1)
    new_range2 = (prev_range2[0] + shift2, prev_range2[1] + shift2)
    
    return (new_range1, new_range2)


def fit_spectrum_adaptive(freq, spectrum, peak1_freq, peak2_freq, verbose=False, width_multiplier=3.0):
    """
    Адаптивный фиттинг спектра с выбором диапазона вокруг пиков
    
    Определяет диапазоны вокруг каждого пика (±width_multiplier * FWHM).
    Если диапазоны перекрываются - фитит двумя лоренцианами вместе.
    Если не перекрываются - фитит каждый пик отдельно.
    
    Parameters:
    -----------
    freq : array-like
        Массив частот (ГГц)
    spectrum : array-like
        Массив амплитуд S-параметра
    peak1_freq : float
        Частота первого пика (ГГц)
    peak2_freq : float
        Частота второго пика (ГГц)
    verbose : bool
        Выводить подробности
    width_multiplier : float
        Множитель ширины для определения диапазона фиттинга (по умолчанию 3)
        
    Returns:
    --------
    fitted_params : dict
        Подогнанные параметры (структура зависит от метода фиттинга)
    fit_quality : dict
        Качество подгонки
    fitted_spectrum : array-like
        Восстановленный спектр (для всего диапазона freq)
    fit_method : str
        Метод фиттинга: 'two_lorentzians' или 'separate'
    """
    # Оценка параметров для обоих пиков
    params1 = estimate_peak_parameters(freq, spectrum, peak1_freq)
    params2 = estimate_peak_parameters(freq, spectrum, peak2_freq)
    
    if verbose:
        print(f"  Оценка пика 1: f={params1['f0']:.6f} ГГц, w={params1['width']:.6f} ГГц")
        print(f"  Оценка пика 2: f={params2['f0']:.6f} ГГц, w={params2['width']:.6f} ГГц")
    
    # Определить диапазоны фиттинга для каждого пика
    range1_min = params1['f0'] - width_multiplier * params1['width']
    range1_max = params1['f0'] + width_multiplier * params1['width']
    range2_min = params2['f0'] - width_multiplier * params2['width']
    range2_max = params2['f0'] + width_multiplier * params2['width']
    
    # Проверка перекрытия диапазонов
    ranges_overlap = not (range1_max < range2_min or range2_max < range1_min)
    
    if ranges_overlap:
        # Диапазоны перекрываются - фитить двумя лоренцианами вместе
        if verbose:
            print(f"  Диапазоны пиков перекрываются → фит двумя лоренцианами")
        
        # Объединенный диапазон
        fit_range_min = min(range1_min, range2_min)
        fit_range_max = max(range1_max, range2_max)
        
        # Маска для диапазона фиттинга
        fit_mask = (freq >= fit_range_min) & (freq <= fit_range_max)
        
        if not np.any(fit_mask):
            if verbose:
                print(f"  ✗ Нет данных в диапазоне фиттинга")
            return None, None, None, 'failed'
        
        freq_fit = freq[fit_mask]
        spectrum_fit = spectrum[fit_mask]
        
        # Подготовка начальных параметров для двойного фиттинга
        initial_params = {
            'f1': params1['f0'],
            'w1': params1['width'],
            'a1': params1['amplitude'],
            'f2': params2['f0'],
            'w2': params2['width'],
            'a2': params2['amplitude'],
            'baseline': (params1['baseline'] + params2['baseline']) / 2.0
        }
        
        # Фиттинг
        fitted_params, fit_quality, fitted_spectrum_local = fit_spectrum_with_two_lorentzians(
            freq_fit, spectrum_fit, initial_params, verbose=verbose
        )
        
        if fitted_params is None:
            return None, None, None, 'failed'
        
        # Восстановить спектр для всего диапазона
        fitted_spectrum_full = two_lorentzians_dip(
            freq,
            fitted_params['f1'], fitted_params['w1'], fitted_params['a1'],
            fitted_params['f2'], fitted_params['w2'], fitted_params['a2'],
            fitted_params['baseline']
        )
        
        return fitted_params, fit_quality, fitted_spectrum_full, 'two_lorentzians'
        
    else:
        # Диапазоны НЕ перекрываются - фитить отдельно
        if verbose:
            print(f"  Диапазоны пиков НЕ перекрываются → отдельные фиты")
        
        # Фиттинг первого пика
        mask1 = (freq >= range1_min) & (freq <= range1_max)
        if not np.any(mask1):
            if verbose:
                print(f"  ✗ Нет данных для пика 1")
            return None, None, None, 'failed'
        
        freq_fit1 = freq[mask1]
        spectrum_fit1 = spectrum[mask1]
        
        fit1_params, fit1_quality, _ = fit_single_lorentzian(
            freq_fit1, spectrum_fit1, params1['f0'], params1['width'], verbose=verbose
        )
        
        # Фиттинг второго пика
        mask2 = (freq >= range2_min) & (freq <= range2_max)
        if not np.any(mask2):
            if verbose:
                print(f"  ✗ Нет данных для пика 2")
            return None, None, None, 'failed'
        
        freq_fit2 = freq[mask2]
        spectrum_fit2 = spectrum[mask2]
        
        fit2_params, fit2_quality, _ = fit_single_lorentzian(
            freq_fit2, spectrum_fit2, params2['f0'], params2['width'], verbose=verbose
        )
        
        if fit1_params is None or fit2_params is None:
            return None, None, None, 'failed'
        
        # Объединить результаты в формат двух пиков
        fitted_params = {
            'f1': fit1_params['f0'],
            'w1': fit1_params['width'],
            'a1': fit1_params['amplitude'],
            'f2': fit2_params['f0'],
            'w2': fit2_params['width'],
            'a2': fit2_params['amplitude'],
            'baseline': (fit1_params['baseline'] + fit2_params['baseline']) / 2.0
        }
        
        # Восстановить спектр для всего диапазона (сумма двух лоренцианов)
        fitted_spectrum_full = two_lorentzians_dip(
            freq,
            fitted_params['f1'], fitted_params['w1'], fitted_params['a1'],
            fitted_params['f2'], fitted_params['w2'], fitted_params['a2'],
            fitted_params['baseline']
        )
        
        # Качество подгонки для полного спектра
        residuals = spectrum - fitted_spectrum_full
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((spectrum - np.mean(spectrum))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        rmse = np.sqrt(np.mean(residuals**2))
        
        fit_quality = {
            'r_squared': r_squared,
            'rmse': rmse,
            'residuals': residuals,
            'fit1_quality': fit1_quality,
            'fit2_quality': fit2_quality
        }
        
        return fitted_params, fit_quality, fitted_spectrum_full, 'separate'


def interactive_fit_verification(freq, spectrum, fitted_params, fit_quality, 
                                 fitted_spectrum, field_value, current_ranges=None):
    """
    Интерактивная проверка результата фиттинга с возможностью корректировки
    
    Показывает график фиттинга и позволяет пользователю:
    - Принять результат (нажать 'y')
    - Указать новые ДИАПАЗОНЫ для фиттинга 4 кликами:
      клик 1 - начало диапазона пика 1
      клик 2 - конец диапазона пика 1
      клик 3 - начало диапазона пика 2
      клик 4 - конец диапазона пика 2
    
    Parameters:
    -----------
    freq : array-like
        Массив частот (ГГц)
    spectrum : array-like
        Экспериментальный спектр
    fitted_params : dict
        Подогнанные параметры
    fit_quality : dict
        Качество подгонки
    fitted_spectrum : array-like or dict
        Восстановленный спектр. Может быть:
        - array: для режима 'combined' (одна линия аппроксимации)
        - dict с ключами {'peak1', 'peak2', 'combined'}: для режима 'separate' 
          (два пика рисуются отдельными линиями)
    field_value : float
        Значение магнитного поля (Э)
    current_ranges : tuple or None
        Текущие диапазоны фиттинга (range1, range2) для отображения
        
    Returns:
    --------
    status : str
        'approved', 'refit', 'skip', или 'quit'
    new_ranges : tuple or None
        Новые диапазоны ((f1_min, f1_max), (f2_min, f2_max)) если пользователь указал их
    """
    # Создание окна
    fig = plt.figure(figsize=(12, 7))
    fig.canvas.manager.set_window_title(f'Проверка фиттинга - Поле {field_value:.2f} Э')
    ax = fig.add_subplot(111)
    
    # Преобразование в дБ для лучшей видимости
    # Избегаем log(0) добавляя малое значение
    epsilon = 1e-10
    spectrum_db = 20 * np.log10(spectrum + epsilon)
    
    # Построение графика в дБ
    ax.plot(freq, spectrum_db, 'o', color='gray', markersize=4, 
           alpha=0.6, label='Экспериментальные данные', zorder=1)
    
    # Проверяем формат fitted_spectrum (может быть array или dict для режима 'separate')
    if isinstance(fitted_spectrum, dict):
        # Режим раздельного фиттинга - рисуем два отдельных пика разными линиями
        fitted_peak1_db = 20 * np.log10(fitted_spectrum['peak1'] + epsilon)
        fitted_peak2_db = 20 * np.log10(fitted_spectrum['peak2'] + epsilon)
        fitted_combined_db = 20 * np.log10(fitted_spectrum['combined'] + epsilon)
        
        ax.plot(freq, fitted_peak1_db, '-', color='blue', linewidth=2.5, 
               label='Аппроксимация пика 1', zorder=2, alpha=0.8)
        ax.plot(freq, fitted_peak2_db, '-', color='orange', linewidth=2.5, 
               label='Аппроксимация пика 2', zorder=2, alpha=0.8)
        ax.plot(freq, fitted_combined_db, '--', color='red', linewidth=1.5, 
               label='Суммарная аппроксимация', zorder=3, alpha=0.6)
    else:
        # Режим комбинированного фиттинга - рисуем одну линию
        fitted_spectrum_db = 20 * np.log10(fitted_spectrum + epsilon)
        ax.plot(freq, fitted_spectrum_db, '-', color='red', linewidth=3, 
               label='Аппроксимация', zorder=2)
    
    # Если есть текущие диапазоны - показать их
    if current_ranges is not None:
        range1, range2 = current_ranges
        ax.axvspan(range1[0], range1[1], alpha=0.2, color='blue', 
                  label=f'Диапазон 1: [{range1[0]:.4f}, {range1[1]:.4f}]')
        ax.axvspan(range2[0], range2[1], alpha=0.2, color='orange', 
                  label=f'Диапазон 2: [{range2[0]:.4f}, {range2[1]:.4f}]')
    
    # Маркеры пиков
    ax.axvline(x=fitted_params['f1'], color='blue', linestyle='--', 
              linewidth=2, alpha=0.7, label=f"Пик 1: {fitted_params['f1']:.6f} ГГц")
    ax.axvline(x=fitted_params['f2'], color='orange', linestyle='--', 
              linewidth=2, alpha=0.7, label=f"Пик 2: {fitted_params['f2']:.6f} ГГц")
    
    # Настройка осей
    ax.set_xlabel('Частота (ГГц)', fontsize=14, fontweight='bold')
    ax.set_ylabel('|S| (дБ)', fontsize=14, fontweight='bold')
    ax.set_title(f'Поле: {field_value:.2f} Э | R²={fit_quality["r_squared"]:.6f} | RMSE={fit_quality["rmse"]:.6f}', 
                fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.4, linestyle='--')
    ax.legend(fontsize=10, loc='best')
    
    # Информационный текст
    info_text = (
        f"ИНСТРУКЦИЯ:\n"
        f"• Нажмите 'y' если фиттинг устраивает\n"
        f"• Кликните 4 раза для выбора диапазонов:\n"
        f"  1) Начало диапазона пика 1\n"
        f"  2) Конец диапазона пика 1\n"
        f"  3) Начало диапазона пика 2\n"
        f"  4) Конец диапазона пика 2\n"
        f"• Нажмите 'n' для пропуска этого поля\n"
        f"• Закройте окно для завершения"
    )
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Параметры фиттинга
    params_text = (
        f"ПАРАМЕТРЫ:\n"
        f"f1={fitted_params['f1']:.6f} ГГц\n"
        f"w1={fitted_params['w1']:.6f} ГГц\n"
        f"a1={fitted_params['a1']:.4f}\n"
        f"f2={fitted_params['f2']:.6f} ГГц\n"
        f"w2={fitted_params['w2']:.6f} ГГц\n"
        f"a2={fitted_params['a2']:.4f}"
    )
    ax.text(0.98, 0.98, params_text, transform=ax.transAxes, 
           fontsize=9, verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
           family='monospace')
    
    plt.tight_layout()
    
    # Обработка взаимодействия
    approved = [False]
    clicked_freqs = []
    skip_field = [False]
    quit_tracking = [False]
    click_markers = []
    
    def on_key(event):
        if event.key == 'y':
            approved[0] = True
            plt.close(fig)
        elif event.key == 'n':
            skip_field[0] = True
            plt.close(fig)
    
    def on_close(event):
        # Если окно закрывается без действий - это quit
        if not approved[0] and not skip_field[0] and len(clicked_freqs) < 4:
            quit_tracking[0] = True
    
    def on_click(event):
        if event.inaxes != ax:
            return
        
        freq_click = event.xdata
        if freq_click is None:
            return
        
        clicked_freqs.append(freq_click)
        
        # Нарисовать вертикальную линию для клика
        click_labels = [
            'Начало дипазона 1',
            'Конец диапазона 1',
            'Начало диапазона 2',
            'Конец диапазона 2'
        ]
        colors = ['blue', 'blue', 'orange', 'orange']
        styles = [':', '-.', ':', '-.']
        
        idx = len(clicked_freqs) - 1
        if idx < 4:
            line = ax.axvline(x=freq_click, color=colors[idx], linestyle=styles[idx], 
                            linewidth=2, alpha=0.9, label=click_labels[idx])
            click_markers.append(line)
            ax.legend(fontsize=9, loc='best')
            fig.canvas.draw()
        
        if len(clicked_freqs) >= 4:
            print(f"\n  → Выбраны диапазоны:")
            print(f"     Пик 1: [{clicked_freqs[0]:.6f}, {clicked_freqs[1]:.6f}] ГГц")
            print(f"     Пик 2: [{clicked_freqs[2]:.6f}, {clicked_freqs[3]:.6f}] ГГц")
            plt.close(fig)
    
    fig.canvas.mpl_connect('key_press_event', on_key)
    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('close_event', on_close)
    
    plt.show()
    
    # Обработка результата
    if quit_tracking[0]:
        return 'quit', None
    
    if skip_field[0]:
        return 'skip', None
    
    if approved[0]:
        return 'approved', None
    
    if len(clicked_freqs) >= 4:
        # Сформировать диапазоны из кликов
        # Убедимся что начало < конца для каждого диапазона
        range1 = (min(clicked_freqs[0], clicked_freqs[1]), 
                  max(clicked_freqs[0], clicked_freqs[1]))
        range2 = (min(clicked_freqs[2], clicked_freqs[3]), 
                  max(clicked_freqs[2], clicked_freqs[3]))
        
        return 'refit', (range1, range2)
    
    # Если окно закрыто другим способом
    return 'quit', None


# =============================================================================
# ОТСЛЕЖИВАНИЕ ПИКОВ ПО ПОЛЯМ
# =============================================================================

def track_peaks_across_fields(data, peak1_init, peak2_init, verbose=True, interactive=False):
    """
    Отслеживание двух пиков по всем значениям магнитного поля
    
    Фитит двумя лоренцианами в заданных диапазонах. Диапазоны выбираются
    пользователем при первом фиттинге, затем автоматически сдвигаются
    следуя за смещением пиков.
    
    Parameters:
    -----------
    data : dict
        Данные S-параметров:
        - 'freq': массив частот (ГГц)
        - 'field': массив полей (Э)
        - 's_param': 2D массив S-параметров
        - 's_type': тип S-параметра ('S21' или 'S12')
    peak1_init : tuple
        Начальная точка первого пика (field, freq)
    peak2_init : tuple
        Начальная точка второго пика (field, freq)
    verbose : bool
        Выводить прогресс
    interactive : bool
        Если True, показывает каждый фиттинг для проверки пользователем
        
    Returns:
    --------
    results : list of dict
        Список результатов для каждого поля
    """
    freq = data['freq']
    field = data['field']
    s_param = data['s_param']
    
    # Найти индекс поля клика
    field_click = (peak1_init[0] + peak2_init[0]) / 2.0
    field_click_idx = np.argmin(np.abs(field - field_click))
    
    if verbose:
        print(f"\nОтслеживание пиков начиная с поля {field[field_click_idx]:.2f} Э")
        print(f"Пик 1: H={peak1_init[0]:.2f} Э, f={peak1_init[1]:.6f} ГГц")
        print(f"Пик 2: H={peak2_init[0]:.2f} Э, f={peak2_init[1]:.6f} ГГц")
        print(f"\nВсего полей для обработки: {len(field)}")
    
    # Извлечь спектр при поле клика
    spectrum_init = np.abs(s_param[field_click_idx, :])
    
    # Начальные диапазоны - по умолчанию ±0.05 ГГц вокруг кликов
    default_width = 0.05
    initial_range1 = (peak1_init[1] - default_width, peak1_init[1] + default_width)
    initial_range2 = (peak2_init[1] - default_width, peak2_init[1] + default_width)
    
    # Фиттинг при поле клика
    if verbose:
        print(f"\n{'='*70}")
        print(f"Фиттинг при поле клика: {field[field_click_idx]:.2f} Э")
        print(f"{'='*70}")
    
    # Цикл фиттинга с проверкой и корректировкой диапазонов
    current_ranges = (initial_range1, initial_range2)
    max_validation_attempts = 3  # Уменьшено с 5 до 3 для предотвращения чрезмерного сужения
    
    for attempt in range(max_validation_attempts):
        # Выполняем фиттинг в текущих диапазонах
        fitted_params, fit_quality, fitted_spectrum, used_ranges = fit_two_lorentzians_in_ranges(
            freq, spectrum_init, current_ranges[0], current_ranges[1], verbose=verbose
        )
        
        if fitted_params is None:
            print(f"✗ ОШИБКА: Не удалось выполнить фиттинг при поле клика")
            return []
        
        # Проверяем корректность диапазонов
        status, adjusted_ranges, diagnostics = validate_and_adjust_ranges(
            fitted_params, current_ranges, centering_threshold=0.20, width_tolerance=0.5  # Более мягкие пороги
        )
        
        if status == 'ok':
            # Диапазоны корректны
            if verbose:
                print(f"  ✓ Диапазоны валидны, фиттинг принят (попытка {attempt + 1})")
            
            # Интерактивная проверка, если режим включен
            if interactive:
                user_status, new_ranges = interactive_fit_verification(
                    freq, spectrum_init, fitted_params, fit_quality, 
                    fitted_spectrum, field[field_click_idx], current_ranges
                )
                
                if user_status == 'approved':
                    if verbose:
                        print(f"  ✓ Фиттинг принят пользователем")
                    break
                elif user_status == 'refit':
                    if verbose:
                        print(f"  ↻ Повторный фиттинг с новыми диапазонами, указанными пользователем...")
                    current_ranges = new_ranges
                    continue
                elif user_status == 'skip':
                    print(f"  ⊘ Поле пропущено пользователем")
                    return []
                elif user_status == 'quit':
                    print(f"  ✗ Отслеживание прервано пользователем")
                    return []
            else:
                # Без интерактивного режима - принимаем автоматически
                break
        
        elif status == 'too_narrow':
            # Диапазоны слишком узкие, останавливаем итерации
            if verbose:
                print(f"  ⚠ Попытка {attempt + 1}: Диапазоны слишком узкие, используем текущий результат")
                if diagnostics['peak1'].get('too_narrow'):
                    print(f"    Пик 1: ширина диапазона {diagnostics['peak1']['range_width']*1000:.2f} МГц < 5 МГц")
                if diagnostics['peak2'].get('too_narrow'):
                    print(f"    Пик 2: ширина диапазона {diagnostics['peak2']['range_width']*1000:.2f} МГц < 5 МГц")
            break
                
        elif status == 'adjust':
            # Нужна корректировка диапазонов
            if verbose:
                print(f"  ⚠ Попытка {attempt + 1}: Диапазоны требуют корректировки")
                if diagnostics['peak1']['needs_adjustment']:
                    print(f"    Пик 1: смещение={diagnostics['peak1']['rel_shift']*100:.1f}%, "
                          f"соотношение ширин={diagnostics['peak1']['width_ratio']:.2f}")
                if diagnostics['peak2']['needs_adjustment']:
                    print(f"    Пик 2: смещение={diagnostics['peak2']['rel_shift']*100:.1f}%, "
                          f"соотношение ширин={diagnostics['peak2']['width_ratio']:.2f}")
            
            # Применяем скорректированные диапазоны
            current_ranges = adjusted_ranges
            
            if verbose:
                print(f"  → Новые диапазоны:")
                if diagnostics['peak1']['needs_adjustment']:
                    print(f"    Диапазон 1: [{current_ranges[0][0]:.6f}, {current_ranges[0][1]:.6f}] ГГц")
                if diagnostics['peak2']['needs_adjustment']:
                    print(f"    Диапазон 2: [{current_ranges[1][0]:.6f}, {current_ranges[1][1]:.6f}] ГГц")
            
            # Продолжаем цикл с новыми диапазонами
            continue
    else:
        # Достигли максимума попыток
        if verbose:
            print(f"⚠ ПРЕДУПРЕЖДЕНИЕ: Достигнут лимит попыток ({max_validation_attempts})")
            print(f"  Используем последний результат фиттинга")

    
    # Список результатов
    results = []
    
    # Сохраняем результат для поля клика
    result_init = {
        'field': field[field_click_idx],
        **fitted_params,
        **fit_quality
    }
    results.append(result_init)
    
    # ==========================================================================
    # ОТСЛЕЖИВАНИЕ ВВЕРХ ПО ПОЛЮ (от клика к большим полям)
    # ==========================================================================
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Отслеживание пиков ВВЕРХ по полю (от {field[field_click_idx]:.1f} Э к {field[-1]:.1f} Э)")
        print(f"{'='*70}")
    
    prev_ranges = current_ranges
    prev_peak_freqs = (fitted_params['f1'], fitted_params['f2'])
    
    for i in range(field_click_idx + 1, len(field)):
        field_value = field[i]
        spectrum = np.abs(s_param[i, :])
        
        # Обновить диапазоны на основе смещения пиков
        current_ranges = update_ranges_by_shift(freq, spectrum, prev_ranges, prev_peak_freqs)
        
        # Цикл фиттинга с проверкой и корректировкой диапазонов
        fit_accepted = False
        max_validation_attempts = 3  # Уменьшено для предотвращения чрезмерного сужения
        
        for attempt in range(max_validation_attempts):
            # Фиттинг в текущих диапазонах
            fitted_params, fit_quality, fitted_spectrum, used_ranges = fit_two_lorentzians_in_ranges(
                freq, spectrum, current_ranges[0], current_ranges[1], verbose=False
            )
            
            if fitted_params is None:
                if verbose:
                    print(f"  ⚠ Фиттинг не удался для поля {field_value:.2f} Э, пропускаем")
                break
            
            # Проверяем корректность диапазонов
            status, adjusted_ranges, diagnostics = validate_and_adjust_ranges(
                fitted_params, current_ranges, centering_threshold=0.20, width_tolerance=0.5
            )
            
            if status == 'ok':
                # Диапазоны корректны
                # Интерактивная проверка, если режим включен
                if interactive:
                    user_status, new_ranges = interactive_fit_verification(
                        freq, spectrum, fitted_params, fit_quality, 
                        fitted_spectrum, field_value, current_ranges
                    )
                    
                    if user_status == 'approved':
                        fit_accepted = True
                        break
                    elif user_status == 'refit':
                        if verbose:
                            print(f"  ↻ Повторный фиттинг для поля {field_value:.2f} Э...")
                        current_ranges = new_ranges
                        continue
                    elif user_status == 'skip':
                        if verbose:
                            print(f"  ⊘ Поле {field_value:.2f} Э пропущено пользователем")
                        break
                    elif user_status == 'quit':
                        if verbose:
                            print(f"  ✗ Отслеживание прервано пользователем")
                        return results
                else:
                    # Без интерактивного режима - принимаем автоматически
                    fit_accepted = True
                    break
            
            elif status == 'too_narrow':
                # Диапазоны слишком узкие, используем текущий результат
                if verbose and attempt == 0:
                    print(f"  ⚠ Поле {field_value:.2f} Э: диапазоны слишком узкие, используем как есть")
                fit_accepted = True
                break
                    
            elif status == 'adjust':
                # Нужна корректировка диапазонов
                if attempt == 0 and verbose:  # Выводим только при первой попытке
                    if diagnostics['peak1']['needs_adjustment'] or diagnostics['peak2']['needs_adjustment']:
                        print(f"  → Поле {field_value:.2f} Э: корректировка диапазонов")
                
                # Применяем скорректированные диапазоны
                current_ranges = adjusted_ranges
                continue
        else:
            # Достигли максимума попыток, но всё равно используем последний результат
            if fitted_params is not None:
                fit_accepted = True
        
        if fit_accepted and fitted_params is not None:
            # Сохранение результатов
            result = {
                'field': field_value,
                **fitted_params,
                **fit_quality
            }
            results.append(result)
            
            # Обновление для следующей итерации
            prev_ranges = current_ranges
            prev_peak_freqs = (fitted_params['f1'], fitted_params['f2'])
            
            # Прогресс
            if verbose and not interactive and (i - field_click_idx) % 10 == 0:
                print(f"  Обработано полей: {i - field_click_idx}/{len(field) - field_click_idx - 1}, "
                      f"текущее поле: {field_value:.2f} Э")
    
    # ==========================================================================
    # ОТСЛЕЖИВАНИЕ ВНИЗ ПО ПОЛЮ (от клика к малым полям)
    # ==========================================================================
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Отслеживание пиков ВНИЗ по полю (от {field[field_click_idx]:.1f} Э к {field[0]:.1f} Э)")
        print(f"{'='*70}")
    
    # Восстанавливаем диапазоны с поля клика
    # Находим диапазоны из первого результата
    # Придется пересчитать из частот и ширин
    first_result = results[0]
    # Восстановим ширины диапазонов из начальных
    range1_width = initial_range1[1] - initial_range1[0]
    range2_width = initial_range2[1] - initial_range2[0]
    prev_ranges = (
        (first_result['f1'] - range1_width/2, first_result['f1'] + range1_width/2),
        (first_result['f2'] - range2_width/2, first_result['f2'] + range2_width/2)
    )
    prev_peak_freqs = (first_result['f1'], first_result['f2'])
    
    for i in range(field_click_idx - 1, -1, -1):
        field_value = field[i]
        spectrum = np.abs(s_param[i, :])
        
        # Обновить диапазоны на основе смещения пиков
        current_ranges = update_ranges_by_shift(freq, spectrum, prev_ranges, prev_peak_freqs)
        
        # Цикл фиттинга с проверкой и корректировкой диапазонов
        fit_accepted = False
        max_validation_attempts = 3  # Уменьшено для предотвращения чрезмерного сужения
        
        for attempt in range(max_validation_attempts):
            # Фиттинг в текущих диапазонах
            fitted_params, fit_quality, fitted_spectrum, used_ranges = fit_two_lorentzians_in_ranges(
                freq, spectrum, current_ranges[0], current_ranges[1], verbose=False
            )
            
            if fitted_params is None:
                if verbose:
                    print(f"  ⚠ Фиттинг не удался для поля {field_value:.2f} Э, пропускаем")
                break
            
            # Проверяем корректность диапазонов
            status, adjusted_ranges, diagnostics = validate_and_adjust_ranges(
                fitted_params, current_ranges, centering_threshold=0.20, width_tolerance=0.5
            )
            
            if status == 'ok':
                # Диапазоны корректны
                # Интерактивная проверка, если режим включен
                if interactive:
                    user_status, new_ranges = interactive_fit_verification(
                        freq, spectrum, fitted_params, fit_quality, 
                        fitted_spectrum, field_value, current_ranges
                    )
                    
                    if user_status == 'approved':
                        fit_accepted = True
                        break
                    elif user_status == 'refit':
                        if verbose:
                            print(f"  ↻ Повторный фиттинг для поля {field_value:.2f} Э...")
                        current_ranges = new_ranges
                        continue
                    elif user_status == 'skip':
                        if verbose:
                            print(f"  ⊘ Поле {field_value:.2f} Э пропущено пользователем")
                        break
                    elif user_status == 'quit':
                        if verbose:
                            print(f"  ✗ Отслеживание прервано пользователем")
                        return results
                else:
                    # Без интерактивного режима - принимаем автоматически
                    fit_accepted = True
                    break
            
            elif status == 'too_narrow':
                # Диапазоны слишком узкие, используем текущий результат
                if verbose and attempt == 0:
                    print(f"  ⚠ Поле {field_value:.2f} Э: диапазоны слишком узкие, используем как есть")
                fit_accepted = True
                break
                    
            elif status == 'adjust':
                # Нужна корректировка диапазонов
                if attempt == 0 and verbose:  # Выводим только при первой попытке
                    if diagnostics['peak1']['needs_adjustment'] or diagnostics['peak2']['needs_adjustment']:
                        print(f"  → Поле {field_value:.2f} Э: корректировка диапазонов")
                
                # Применяем скорректированные диапазоны
                current_ranges = adjusted_ranges
                continue
        else:
            # Достигли максимума попыток, но всё равно используем последний результат
            if fitted_params is not None:
                fit_accepted = True
        
        if fit_accepted and fitted_params is not None:
            # Сохранение результатов
            result = {
                'field': field_value,
                **fitted_params,
                **fit_quality
            }
            # Вставляем в начало списка, чтобы сохранить порядок по полю
            results.insert(0, result)
            
            # Обновление для следующей итерации
            prev_ranges = current_ranges
            prev_peak_freqs = (fitted_params['f1'], fitted_params['f2'])
            
            # Прогресс
            if verbose and not interactive and (field_click_idx - i) % 10 == 0:
                print(f"  Обработано полей: {field_click_idx - i}/{field_click_idx}, "
                      f"текущее поле: {field_value:.2f} Э")
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"✓ Отслеживание завершено")
        print(f"  Успешно обработано полей: {len(results)}/{len(field)}")
        print(f"{'='*70}")
    
    return results


# =============================================================================
# СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# =============================================================================

def save_peak_tracking_results(results, results_dir, filename='peak_parameters.csv'):
    """
    Сохранить результаты отслеживания пиков в CSV файл
    
    Parameters:
    -----------
    results : list of dict
        Результаты отслеживания пиков
    results_dir : str
        Директория для сохранения
    filename : str
        Имя файла
        
    Returns:
    --------
    filepath : str
        Полный путь к сохраненному файлу
    """
    filepath = os.path.join(results_dir, filename)
    
    # Заголовок
    header = "Field(Э),Freq1(ГГц),Width1(ГГц),Amp1,Freq2(ГГц),Width2(ГГц),Amp2,Baseline,R²,RMSE\n"
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(header)
        
        for result in results:
            line = (f"{result['field']:.4f},"
                   f"{result['f1']:.8f},"
                   f"{result['w1']:.8f},"
                   f"{result['a1']:.8f},"
                   f"{result['f2']:.8f},"
                   f"{result['w2']:.8f},"
                   f"{result['a2']:.8f},"
                   f"{result['baseline']:.8f},"
                   f"{result['r_squared']:.8f},"
                   f"{result['rmse']:.8f}\n")
            f.write(line)
    
    print(f"\n✓ Результаты сохранены в {filepath}")
    
    return filepath


def save_peak_tracking_summary(results, results_dir, filename='peak_tracking_summary.txt'):
    """
    Сохранить сводную информацию об отслеживании пиков
    
    Parameters:
    -----------
    results : list of dict
        Результаты отслеживания пиков
    results_dir : str
        Директория для сохранения
    filename : str
        Имя файла
    """
    filepath = os.path.join(results_dir, filename)
    
    # Вычисление статистики
    fields = [r['field'] for r in results]
    f1_values = [r['f1'] for r in results]
    f2_values = [r['f2'] for r in results]
    w1_values = [r['w1'] for r in results]
    w2_values = [r['w2'] for r in results]
    r2_values = [r['r_squared'] for r in results]
    rmse_values = [r['rmse'] for r in results]
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("РЕЗУЛЬТАТЫ ОТСЛЕЖИВАНИЯ СОБСТВЕННЫХ ЧАСТОТ СИСТЕМЫ\n")
        f.write("=" * 100 + "\n\n")
        
        f.write(f"Количество обработанных полей: {len(results)}\n")
        f.write(f"Диапазон полей: {min(fields):.2f} - {max(fields):.2f} Э\n\n")
        
        f.write("СТАТИСТИКА ПЕРВОГО ПИКА:\n")
        f.write("-" * 50 + "\n")
        f.write(f"Частота (f1):\n")
        f.write(f"  Минимум:  {min(f1_values):.6f} ГГц\n")
        f.write(f"  Максимум: {max(f1_values):.6f} ГГц\n")
        f.write(f"  Среднее:  {np.mean(f1_values):.6f} ГГц\n")
        f.write(f"Ширина (w1):\n")
        f.write(f"  Минимум:  {min(w1_values):.6f} ГГц\n")
        f.write(f"  Максимум: {max(w1_values):.6f} ГГц\n")
        f.write(f"  Среднее:  {np.mean(w1_values):.6f} ГГц\n\n")
        
        f.write("СТАТИСТИКА ВТОРОГО ПИКА:\n")
        f.write("-" * 50 + "\n")
        f.write(f"Частота (f2):\n")
        f.write(f"  Минимум:  {min(f2_values):.6f} ГГц\n")
        f.write(f"  Максимум: {max(f2_values):.6f} ГГц\n")
        f.write(f"  Среднее:  {np.mean(f2_values):.6f} ГГц\n")
        f.write(f"Ширина (w2):\n")
        f.write(f"  Минимум:  {min(w2_values):.6f} ГГц\n")
        f.write(f"  Максимум: {max(w2_values):.6f} ГГц\n")
        f.write(f"  Среднее:  {np.mean(w2_values):.6f} ГГц\n\n")
        
        f.write("КАЧЕСТВО ПОДГОНКИ:\n")
        f.write("-" * 50 + "\n")
        f.write(f"R² (коэффициент детерминации):\n")
        f.write(f"  Минимум:  {min(r2_values):.6f}\n")
        f.write(f"  Максимум: {max(r2_values):.6f}\n")
        f.write(f"  Среднее:  {np.mean(r2_values):.6f}\n")
        f.write(f"RMSE (среднеквадратичная ошибка):\n")
        f.write(f"  Минимум:  {min(rmse_values):.6f}\n")
        f.write(f"  Максимум: {max(rmse_values):.6f}\n")
        f.write(f"  Среднее:  {np.mean(rmse_values):.6f}\n")
    
    print(f"✓ Сводка сохранена в {filepath}")
