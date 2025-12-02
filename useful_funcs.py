import scipy as sp
import numpy as np
import config_physics

def find_peak_flexible(freqs, s_values, expected_freq, expected_width, expected_prominence, 
                       search_window=None, peak_type=None):
    """
    Гибкий поиск пика по примерным параметрам.
    
    Parameters:
    -----------
    freqs : array
        Массив частот (ГГц)
    s_values : array
        Массив значений S-параметра (должен быть по модулю)
    expected_freq : float
        Ожидаемая частота пика (ГГц)
    expected_width : float
        Примерная ширина пика (ГГц)
    expected_prominence : float
        Примерная высота/глубина пика (в единицах S-параметра)
    search_window : float, optional
        Ширина окна поиска вокруг expected_freq (ГГц). 
        По умолчанию = expected_width * 3
    peak_type : str, optional
        'maximum' или 'minimum'. Если None, берется из config_physics.PEAK_TYPE
    
    Returns:
    --------
    dict : {
        'freq': float - частота найденного пика,
        'magnitude': float - амплитуда S-параметра в пике,
        'prominence': float - prominence пика,
        'width': float - ширина пика (ГГц),
        'method': str - метод нахождения ('find_peaks' или 'extremum')
    }
    """
    if peak_type is None:
        peak_type = config_physics.PEAK_TYPE
    
    if search_window is None:
        search_window = expected_width * 3
    
    # Определяем окрестность для поиска
    freq_range = (expected_freq - search_window/2, expected_freq + search_window/2)
    freq_indices = np.where((freqs >= freq_range[0]) & (freqs <= freq_range[1]))[0]
    
    if len(freq_indices) == 0:
        raise ValueError(f"No data points in search window [{freq_range[0]:.4f}, {freq_range[1]:.4f}] GHz")
    
    # Извлекаем данные в окрестности
    local_freqs = freqs[freq_indices]
    local_s_values = s_values[freq_indices]

    if peak_type == 'minimum':
        local_s_values = -local_s_values  # Инвертируем для поиска минимумов
    
    # Вычисляем шаг по частоте
    freq_step = local_freqs[1] - local_freqs[0] if len(local_freqs) > 1 else 0.001
    
    # Преобразуем параметры в количество точек (с защитой от нуля)
    width_points = max(1, int(expected_width / freq_step / 2))  # Делим на 2 для мягкости
    
    # Пытаемся найти пик с помощью find_peaks с мягкими критериями
    found_peaks = sp.signal.find_peaks(
        local_s_values,
        prominence=expected_prominence * 0.5,  # Снижаем требование
        width=width_points
    )
    
    # Если find_peaks нашел пики
    if len(found_peaks[0]) > 0:
        # Выбираем самый яркий пик (с максимальной prominence)
        if len(found_peaks[0]) == 1:
            peak_idx = found_peaks[0][0]
            idx_in_result = 0
        else:
            # Находим индекс пика с максимальной prominence
            idx_in_result = np.argmax(found_peaks[1]['prominences'])
            peak_idx = found_peaks[0][idx_in_result]
        
        peak_freq = local_freqs[peak_idx]
        peak_magnitude = local_s_values[peak_idx] if peak_type == 'maximum' else -local_s_values[peak_idx]
        peak_prominence = found_peaks[1]['prominences'][idx_in_result]
        peak_width = found_peaks[1]['widths'][idx_in_result] * freq_step
        
        return {
            'freq': peak_freq,
            'magnitude': peak_magnitude,
            'prominence': peak_prominence,
            'width': peak_width,
            'method': 'find_peaks'
        }
    
    # Если find_peaks не нашел - ищем экстремум в окрестности
    if peak_type == 'minimum':
        peak_idx = np.argmin(local_s_values)
    else:  # 'maximum'
        peak_idx = np.argmax(local_s_values)
    
    peak_freq = local_freqs[peak_idx]
    peak_magnitude = local_s_values[peak_idx]
    
    # Оцениваем prominence вручную (разница между пиком и средним значением боковых частей)
    left_baseline = np.median(local_s_values[:max(1, peak_idx)]) if peak_idx > 0 else peak_magnitude
    right_baseline = np.median(local_s_values[min(len(local_s_values)-1, peak_idx+1):]) if peak_idx < len(local_s_values)-1 else peak_magnitude
    baseline = (left_baseline + right_baseline) / 2
    peak_prominence = abs(peak_magnitude - baseline)
    
    # Используем исходную оценку ширины
    peak_width = expected_width
    
    return {
        'freq': peak_freq,
        'magnitude': peak_magnitude,
        'prominence': peak_prominence,
        'width': peak_width,
        'method': 'extremum'
    }

def estimate_cavity_params(res_magnitude, resonance_freq, cavity_width, plato):
    
    con = np.abs(res_magnitude - plato)

    kappa = con * cavity_width / 2
    beta = cavity_width / 2 * (1 - con)

    return {'kappa': kappa,
            'beta': beta,
            'resonance_freq': resonance_freq,
            'plato': plato,
            'res_magnitude': res_magnitude}

def fit_cavity_response(freqs, s_average, initial_params):
    p0 = [initial_params['kappa'], initial_params['beta'],
          initial_params['resonance_freq'], initial_params['plato']]

    popt, pcov = sp.optimize.curve_fit(cavity_model, freqs, s_average, p0=p0)

    fitted_params = {
        'kappa': popt[0],
        'beta': popt[1],
        'resonance_freq': popt[2],
        'plato': popt[3]
    }
    fitted_params['res_magnitude'] = cavity_model(fitted_params['resonance_freq'],
                                                  fitted_params['kappa'],
                                                  fitted_params['beta'],
                                                  fitted_params['resonance_freq'],
                                                  fitted_params['plato'])

    return fitted_params

def cavity_model(f, kappa, beta, f0, plato):
        delta_f = f - f0
        response = plato + kappa / np.sqrt(delta_f**2 + (kappa + beta)**2)
        return response

def estimate_peak_width(data, peak):
    freqs = data['freq']
    s_values = data['s_param']
    peak_freq = peak['freq']
    peak_index = np.argmin(np.abs(freqs - peak_freq))
    peak_magnitude = (peak['s_param'] - peak['plato'])
    half_max = peak_magnitude / 2 + peak['plato']

    left_part = s_values[:peak_index]
    right_part = s_values[peak_index:]

    left_indices = np.where(left_part <= half_max)[0] if peak['peak_type'] == 'minimum' else np.where(left_part >= half_max)[0]
    right_indices = np.where(right_part <= half_max)[0] if peak['peak_type'] == 'minimum' else np.where(right_part >= half_max)[0]

    if left_indices.size == 0 or right_indices.size == 0:
        return None
    left_half_max_index = left_indices[0]
    right_half_max_index = peak_index + right_indices[-1]

    fwhm = freqs[right_half_max_index] - freqs[left_half_max_index]
    return fwhm

def fano_model(f, f0, gamma, q, a, b):
    epsilon = (f - f0) / (gamma / 2)
    fano_line_normilized = ((q + epsilon)**2) / (1 + epsilon**2) / (1 + q**2)
    fano_line = a * fano_line_normilized + b
    return fano_line

def calibrate_magnon_frequency(fields, magnon_freqs_experimental):
    """
    Калибрует частоту магнонов по гиромагнитному соотношению
    
    Использует экспериментальную частоту при минимальном поле для калибровки,
    затем рассчитывает линейную зависимость частоты от поля по гиромагнитному соотношению.
    
    Parameters:
    -----------
    fields : array
        Массив значений магнитного поля (Oe)
    magnon_freqs_experimental : array
        Экспериментально полученные частоты магнонов (GHz)
        
    Returns:
    --------
    magnon_freqs_calibrated : array
        Откалиброванные частоты магнонов по гиромагнитному соотношению (GHz)
    offset : float
        Найденное смещение частоты (GHz)
    """
    # Берем минимальное поле и соответствующую частоту для калибровки
    min_field_idx = np.argmin(fields)
    min_field = fields[min_field_idx]
    freq_at_min_field = magnon_freqs_experimental[min_field_idx]
    
    # Гиромагнитное отношение из конфига (ГГц/Э)
    gamma_g = config_physics.GYROMAGNETIC_RATIO
    
    # Рассчитываем смещение (offset)
    # f_magnon = gamma_g * H + offset
    # offset = f_magnon - gamma_g * H
    offset = freq_at_min_field - gamma_g * min_field
    
    # Рассчитываем откалиброванные частоты для всех полей
    magnon_freqs_calibrated = gamma_g * fields + offset
    
    print(f"\n{'='*60}")
    print("MAGNON FREQUENCY CALIBRATION")
    print(f"{'='*60}")
    print(f"Gyromagnetic ratio: {gamma_g:.6f} GHz/Oe")
    print(f"Calibration point: H = {min_field:.2f} Oe, f = {freq_at_min_field:.6f} GHz")
    print(f"Frequency offset: {offset:.6f} GHz")
    print(f"{'='*60}\n")
    
    return magnon_freqs_calibrated, offset
