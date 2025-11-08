"""
Функции фиттинга данных антикроссинга

Содержит функции для подгонки экспериментальных данных к теоретическим моделям
"""

import numpy as np
from scipy.optimize import curve_fit

import config_physics
import models


# =============================================================================
# ФИТТИНГ СПЕКТРОВ
# =============================================================================

def fit_spectrum_one_mode(freq, s_param_amplitude, field_value, initial_params, param_bounds):
    """
    Фиттинг амплитуды S-параметра для одного значения поля
    
    Фиксированные параметры:
    - wc, kappa, beta (параметры резонатора)
    - H0, w0, gamma_g (параметры магнонов - резонансная частота)
    
    Параметры для фиттинга:
    - J (когерентная связь)
    - gamma (внешние потери магнонов)
    - alpha (внутренние потери магнонов)
    
    Вычисляемые параметры:
    - Gamma = sqrt(kappa * gamma) (диссипативная связь)
    
    Parameters:
    -----------
    freq : array-like
        Массив частот (ГГц)
    s_param_amplitude : array-like
        Амплитуда S-параметра |S| в линейной шкале
    field_value : float
        Значение магнитного поля для данного спектра (Э)
    initial_params : dict
        Начальные значения параметров для фиттинга
    param_bounds : dict
        Границы параметров для фиттинга
        
    Returns:
    --------
    fitted_params : dict
        Подогнанные параметры модели
    fit_quality : dict
        Метрики качества подгонки (R², среднеквадратичная ошибка и т.д.)
        
    Использует scipy.optimize.curve_fit для оптимизации параметров
    """
    # Список параметров для фиттинга (только связь и потери магнонов)
    param_names = ['J', 'gamma', 'alpha']
    
    # Начальные значения и границы
    p0 = [initial_params[name] for name in param_names]
    bounds_lower = [param_bounds[name][0] for name in param_names]
    bounds_upper = [param_bounds[name][1] for name in param_names]
    
    # Фиксированные параметры резонатора
    wc = initial_params['wc']
    kappa = initial_params['kappa']
    beta = initial_params['beta']
    
    # Фиксированные параметры магнонов (резонансная частота)
    H0 = initial_params['H0']
    w0 = initial_params['w0']
    gamma_g = initial_params.get('gamma_g', config_physics.GYROMAGNETIC_RATIO)
    s_type = initial_params.get('s_type', 'S21')
    
    # Нормализация данных для улучшения численной стабильности
    s_mean = np.mean(s_param_amplitude)
    s_std = np.std(s_param_amplitude)
    if s_std == 0:
        s_std = 1.0
    s_param_normalized = (s_param_amplitude - s_mean) / s_std
    
    # Функция-обертка для модели (принимает развернутые параметры)
    def model_wrapper(freq_input, J, gamma, alpha):
        # Вычисляем диссипативную связь по формуле
        Gamma = models.calculate_dissipative_coupling(kappa, gamma)
        
        params_dict = {
            'wc': wc,
            'kappa': kappa,
            'beta': beta,
            'J': J,
            'Gamma': Gamma,
            'gamma': gamma,
            'alpha': alpha,
            'H0': H0,
            'w0': w0,
            'gamma_g': gamma_g
        }
        # Передаем field_value напрямую как скаляр
        s_complex = models.anticrossing_one_mode_model(freq_input, field_value, params_dict, s_type)
        s_abs = np.abs(s_complex)
        # Нормализуем выход модели так же, как данные
        return (s_abs - np.mean(s_abs)) / np.std(s_abs)
    
    try:
        # Выполнение фиттинга с улучшенными параметрами на нормализованных данных
        popt, pcov = curve_fit(
            model_wrapper,
            freq,
            s_param_normalized,  # Используем нормализованные данные
            p0=p0,
            bounds=(bounds_lower, bounds_upper),
            method='trf',  # Trust Region Reflective - более робастный метод
            maxfev=50000,  # Увеличено количество итераций
            ftol=1e-10,    # Более строгий критерий сходимости для функции
            xtol=1e-10,    # Более строгий критерий сходимости для параметров
            diff_step=1e-8,  # Шаг для численного дифференцирования
            loss='soft_l1',  # Робастная функция потерь, менее чувствительна к выбросам
            verbose=0      # Без вывода подробностей
        )
        
        # Сохранение результатов
        fitted_params = {name: popt[i] for i, name in enumerate(param_names)}
        
        # Вычисляем диссипативную связь из подогнанного gamma
        fitted_params['Gamma'] = models.calculate_dissipative_coupling(kappa, fitted_params['gamma'])
        
        # Добавляем фиксированные параметры резонатора
        fitted_params['wc'] = wc
        fitted_params['kappa'] = kappa
        fitted_params['beta'] = beta
        
        # Добавляем фиксированные параметры магнонов
        fitted_params['H0'] = H0
        fitted_params['w0'] = w0
        fitted_params['gamma_g'] = gamma_g
        fitted_params['field'] = field_value
        
        # Вычисление качества подгонки
        s_fitted = models.anticrossing_one_mode_model(freq, field_value, fitted_params, s_type)
        s_fitted_amplitude = np.abs(s_fitted)
        
        # R² коэффициент
        ss_res = np.sum((s_param_amplitude - s_fitted_amplitude) ** 2)
        ss_tot = np.sum((s_param_amplitude - np.mean(s_param_amplitude)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Среднеквадратичная ошибка
        rmse = np.sqrt(np.mean((s_param_amplitude - s_fitted_amplitude) ** 2))
        
        fit_quality = {
            'r_squared': r_squared,
            'rmse': rmse,
            'residual_sum_squares': ss_res
        }
        
        return fitted_params, fit_quality
        
    except Exception as e:
        # Попытка с расширенными границами (±50% вместо ±20%)
        try:
            print(f"  Первая попытка не удалась для поля {field_value:.1f} Э, пробуем с расширенными границами...")
            
            # Расширяем границы в 2.5 раза
            bounds_lower_wide = [b * 0.5 for b in bounds_lower]
            bounds_upper_wide = [b * 1.5 for b in bounds_upper]
            
            popt, pcov = curve_fit(
                model_wrapper,
                freq,
                s_param_normalized,
                p0=p0,
                bounds=(bounds_lower_wide, bounds_upper_wide),
                method='trf',
                maxfev=50000,
                ftol=1e-10,
                xtol=1e-10,
                diff_step=1e-8,
                loss='soft_l1',
                verbose=0
            )
            
            # Сохранение результатов
            fitted_params = {name: popt[i] for i, name in enumerate(param_names)}
            fitted_params['Gamma'] = models.calculate_dissipative_coupling(kappa, fitted_params['gamma'])
            fitted_params['wc'] = wc
            fitted_params['kappa'] = kappa
            fitted_params['beta'] = beta
            fitted_params['H0'] = H0
            fitted_params['w0'] = w0
            fitted_params['gamma_g'] = gamma_g
            fitted_params['field'] = field_value
            
            # Вычисление качества подгонки
            s_fitted = models.anticrossing_one_mode_model(freq, field_value, fitted_params, s_type)
            s_fitted_amplitude = np.abs(s_fitted)
            
            ss_res = np.sum((s_param_amplitude - s_fitted_amplitude) ** 2)
            ss_tot = np.sum((s_param_amplitude - np.mean(s_param_amplitude)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            rmse = np.sqrt(np.mean((s_param_amplitude - s_fitted_amplitude) ** 2))
            
            fit_quality = {
                'r_squared': r_squared,
                'rmse': rmse,
                'residual_sum_squares': ss_res
            }
            
            print(f"  ✓ Фиттинг успешен с расширенными границами")
            return fitted_params, fit_quality
            
        except Exception as e2:
            print(f"  ✗ Ошибка при фиттинге для поля {field_value:.1f} Э: {e2}")
            return None, None


def fit_all_spectra(data, initial_params, param_bounds):
    """
    Аппроксимация спектров при всех значениях поля
    
    Parameters:
    -----------
    data : dict
        Словарь с данными:
        - 'freq': массив частот (ГГц)
        - 'field': массив полей (Э)
        - 's_param': 2D массив S-параметров
        - 's_type': тип S-параметра ('S21' или 'S12')
    initial_params : dict
        Начальные значения параметров для фиттинга
    param_bounds : dict
        Границы параметров для фиттинга
        
    Returns:
    --------
    fitting_results : list of dict
        Список результатов фиттинга для каждого поля
        Каждый элемент содержит:
        - 'field': значение поля
        - 'wc', 'kappa', 'beta', 'J', 'Gamma', 'gamma', 'alpha': подогнанные параметры
        - 'r_squared': коэффициент детерминации
        - 'rmse': среднеквадратичная ошибка
    fitted_data : dict
        Словарь с восстановленными данными (той же структуры что и data)
        
    Выполняет фиттинг для каждого значения поля и возвращает результаты
    """
    # Извлечение данных
    freq = data['freq']
    field = data['field']
    s_param = data['s_param']
    s_type = data['s_type']
    
    # Список для хранения результатов
    fitting_results = []
    
    # Массив для восстановленных S-параметров
    s_param_fitted = np.zeros_like(s_param, dtype=complex)
    
    print(f"\nФиттинг {len(field)} спектров...")
    
    # Фиттинг для каждого значения поля
    for i, field_value in enumerate(field):
        # Амплитуда S-параметра для данного поля
        s_amplitude = np.abs(s_param[i, :])
        
        # Выполнение фиттинга
        fitted_params, fit_quality = fit_spectrum_one_mode(
            freq, s_amplitude, field_value, initial_params, param_bounds
        )
        
        if fitted_params is not None:
            # Сохранение результатов
            fitting_results.append({
                'field': field_value,
                **fitted_params,
                **fit_quality
            })
            
            # Восстановление S-параметра для данного поля
            s_param_fitted[i, :] = models.anticrossing_one_mode_model(
                freq, field_value, fitted_params, s_type
            )
            
            # Прогресс
            if (i + 1) % 10 == 0:
                print(f"  Обработано {i + 1}/{len(field)} спектров...")
    
    print(f"✓ Фиттинг завершен. Успешно обработано {len(fitting_results)}/{len(field)} спектров")
    
    # Создание словаря с восстановленными данными
    fitted_data = {
        'freq': freq,
        'field': field,
        's_param': s_param_fitted,
        's_type': s_type
    }
    
    return fitting_results, fitted_data


# =============================================================================
# ФИТТИНГ РЕЗОНАТОРА
# =============================================================================

def estimate_cavity_parameters_from_interactive(interactive_params, data, fit_region, from_formula=False):
    """
    Извлечь начальные параметры резонатора из интерактивно выбранных данных
    
    Parameters:
    -----------
    interactive_params : dict
        Интерактивно выбранные параметры:
        - cavity_frequency: частота резонатора (ГГц)
        - cavity_width: ширина резонатора (freq_min, freq_max)
    data : dict
        Данные S-параметров
    fit_region : dict
        Диапазон для фиттинга: {'freq_range': (f_min, f_max), 'field_range': (h_min, h_max)}
    from_formula : bool
        Если True, оценивает kappa и beta по точным формулам резонатора:
        kappa = (width*sqrt(3)*(1-S0))/(2*sqrt(1-4*S0^2))
        beta = (width*sqrt(3)*S0)/(2*sqrt(1-4*S0^2))
        где S0 - значение S-параметра в пике, width - ширина на полувысоте
        
    Returns:
    --------
    initial_params : dict
        Начальные параметры для фиттинга резонатора:
        - wc: частота резонатора (ГГц)
        - kappa: внешние потери резонатора (ГГц)
        - beta: внутренние потери резонатора (ГГц)
    """
    import data_io
    
    # Частота резонатора из интерактивного выбора или из диапазона фиттинга
    cavity_freq = interactive_params.get('cavity_frequency')
    if cavity_freq is None:
        # Если не выбрана интерактивно, используем центр диапазона фиттинга
        freq_min, freq_max = fit_region['freq_range']
        wc = (freq_min + freq_max) / 2.0
        print(f"  Частота резонатора не выбрана, используется центр диапазона: {wc:.6f} ГГц")
    else:
        wc = cavity_freq
    
    # Оценка kappa и beta
    if from_formula:
        # Извлечь данные в диапазоне фиттинга
        filtered_data = data_io.filter_data_by_range(data, fit_region['field_range'], fit_region['freq_range'])
        freq = filtered_data['freq']
        s_param = filtered_data['s_param']
        
        # Усреднить по полю
        s_param_avg = np.mean(np.abs(s_param), axis=0)
        
        # Найти резонанс (МИНИМУМ для проходного режима - провал!)
        peak_idx = np.argmin(s_param_avg)
        S0 = s_param_avg[peak_idx]
        peak_freq = freq[peak_idx]
        
        # Уточнить частоту резонатора по минимуму
        if cavity_freq is None:
            wc = peak_freq
            print(f"  Частота резонатора уточнена по провалу (минимуму): {wc:.6f} ГГц")
        
        # Найти ширину на полувысоте (FWHM)
        # Для провала: полувысота = 1/2 в линейных единицах
        half_height = 0.5
        
        # Найти точки, где сигнал пересекает полувысоту
        # Для провала: ищем где сигнал НИЖЕ полувысоты
        below_half = s_param_avg <= half_height
        
        # Найти левую и правую границы провала
        # Левая граница: первый индекс слева от минимума, где сигнал <= half_height
        left_indices = np.where(below_half[:peak_idx])[0]
        # Правая граница: последний индекс справа от минимума, где сигнал <= half_height
        right_indices = np.where(below_half[peak_idx:])[0] + peak_idx
        
        if len(left_indices) > 0 and len(right_indices) > 0:
            left_idx = left_indices[0]   # Первый индекс слева, где начинается провал
            right_idx = right_indices[-1]  # Последний индекс справа, где заканчивается провал
            width = np.abs(freq[right_idx] - freq[left_idx])
        else:
            # Если не удалось найти полувысоту, используем интерактивную ширину
            cavity_w = interactive_params.get('cavity_width')
            if cavity_w is not None:
                freq_min, freq_max = cavity_w
                width = np.abs(freq_max - freq_min)
            else:
                # Оценка ширины из диапазона фиттинга (10% от диапазона)
                freq_min, freq_max = fit_region['freq_range']
                width = (freq_max - freq_min) * 0.1
            print(f"  Не удалось найти ширину на полувысоте, используется оценка: {width:.6f} ГГц")
        
        # Вычисление kappa и beta по формулам
        # kappa = (width*sqrt(3)*(1-S0))/(2*sqrt(1-4*S0^2))
        # beta = (width*sqrt(3)*S0)/(2*sqrt(1-4*S0^2))
        
        # Проверка условия для формул: |S0| < 0.5 (иначе знаменатель становится мнимым)
        if np.abs(4 * S0**2 - 1) < 1e-10:
            print(f"  ⚠ S0 = {S0:.6f} слишком близко к критическому значению 0.5")
            print(f"  Используется простая оценка: kappa ≈ 0.9*width, beta ≈ 0.1*width")
            kappa_estimate = width * 0.9
            beta_estimate = width * 0.1
        else:
            sqrt_3 = np.sqrt(3)
            denominator = 2 * np.sqrt(np.abs(1 - 4 * S0**2))
            
            kappa_estimate = (width * sqrt_3 * (1 - S0)) / denominator
            beta_estimate = (width * sqrt_3 * S0) / denominator
            
            print(f"  Параметры резонатора по формулам:")
            print(f"  - S0 (пик) = {S0:.6f}")
            print(f"  - width (FWHM) = {width:.6f} ГГц")
        
    else:
        # Простая оценка (старый метод)
        cavity_w = interactive_params.get('cavity_width')
        if cavity_w is not None:
            freq_min, freq_max = cavity_w
            width = np.abs(freq_max - freq_min)
            # Оценка: полная ширина на полувысоте ≈ (kappa + beta)
            # Предполагаем, что kappa >> beta, поэтому kappa ≈ width
            kappa_estimate = width * 0.9  # 90% ширины - внешние потери
            beta_estimate = width * 0.1   # 10% ширины - внутренние потери
        else:
            # Оценка ширины из диапазона фиттинга (10% от диапазона)
            freq_min, freq_max = fit_region['freq_range']
            width = (freq_max - freq_min) * 0.1
            kappa_estimate = width * 0.9
            beta_estimate = width * 0.1
            print(f"  Ширина резонатора не выбрана, оценка из диапазона: {width:.6f} ГГц")
    
    initial_params = {
        'wc': wc,
        'kappa': kappa_estimate,
        'beta': beta_estimate
    }
    
    return initial_params


def fit_cavity_only(data, fit_region, initial_params, param_bounds=None):
    """
    Фиттинг отдельно резонатора (без магнонов) в области вдали от антикроссинга
    
    Parameters:
    -----------
    data : dict
        Данные S-параметров:
        - 'freq': массив частот (ГГц)
        - 'field': массив полей (Э)
        - 's_param': 2D массив S-параметров
        - 's_type': тип S-параметра ('S21' или 'S12')
    fit_region : dict
        Диапазон для фиттинга: {'freq_range': (f_min, f_max), 'field_range': (h_min, h_max)}
    initial_params : dict
        Начальные параметры: wc, kappa, beta
    param_bounds : dict or None
        Границы параметров. Если None, используются широкие границы (0.1x - 10x)
        
    Returns:
    --------
    fitted_params : dict
        Подогнанные параметры резонатора
    fit_quality : dict
        Метрики качества подгонки
    fitted_spectrum : dict
        Восстановленный спектр в диапазоне фиттинга
    """
    import data_io
    
    # Фильтрация данных по указанному диапазону
    freq_range = fit_region['freq_range']
    field_range = fit_region['field_range']
    
    filtered_data = data_io.filter_data_by_range(data, field_range, freq_range)
    
    freq = filtered_data['freq']
    field = filtered_data['field']
    s_param = filtered_data['s_param']
    s_type = filtered_data.get('s_type', 'S21')
    
    # Усредняем по полю (резонатор не зависит от поля в области вдали от антикроссинга)
    s_param_avg = np.mean(np.abs(s_param), axis=0)
    
    # Устанавливаем границы параметров
    # Используем ШИРОКИЕ границы, т.к. начальные оценки могут быть неточными
    if param_bounds is None:
        param_bounds = {
            'wc': (initial_params['wc'] * 0.8, initial_params['wc'] * 1.2),
            'kappa': (max(0.001, initial_params['kappa'] * 0.1), initial_params['kappa'] * 10.0),
            'beta': (max(0.0001, initial_params['beta'] * 0.1), initial_params['beta'] * 10.0)
        }
    
    # Параметры для фиттинга
    param_names = ['wc', 'kappa', 'beta']
    p0 = [initial_params[name] for name in param_names]
    bounds_lower = [param_bounds[name][0] for name in param_names]
    bounds_upper = [param_bounds[name][1] for name in param_names]
    
    # Функция-обертка для модели резонатора
    def model_wrapper(freq_input, wc, kappa, beta):
        s_complex = models.cavity_only_model(freq_input, wc, kappa, beta)
        return np.abs(s_complex)
    
    try:
        # Выполнение фиттинга
        popt, pcov = curve_fit(
            model_wrapper,
            freq,
            s_param_avg,
            p0=p0,
            bounds=(bounds_lower, bounds_upper),
            method='trf',
            maxfev=10000,
            ftol=1e-10,
            xtol=1e-10,
            loss='soft_l1'
        )
        
        # Сохранение результатов
        fitted_params = {name: popt[i] for i, name in enumerate(param_names)}
        
        # Вычисление качества подгонки
        s_fitted = model_wrapper(freq, *popt)
        
        # R² коэффициент
        ss_res = np.sum((s_param_avg - s_fitted) ** 2)
        ss_tot = np.sum((s_param_avg - np.mean(s_param_avg)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Среднеквадратичная ошибка
        rmse = np.sqrt(np.mean((s_param_avg - s_fitted) ** 2))
        
        fit_quality = {
            'r_squared': r_squared,
            'rmse': rmse
        }
        
        # Восстановленный спектр
        fitted_spectrum = {
            'freq': freq,
            'field': field,
            's_param_avg': s_param_avg,
            's_param_fitted': s_fitted,
            's_type': s_type
        }
        
        print(f"\n✓ Фиттинг резонатора завершен успешно")
        print(f"  wc = {fitted_params['wc']:.6f} ГГц")
        print(f"  kappa = {fitted_params['kappa']:.6f} ГГц")
        print(f"  beta = {fitted_params['beta']:.6f} ГГц")
        print(f"  R² = {r_squared:.6f}")
        print(f"  RMSE = {rmse:.6f}")
        
        return fitted_params, fit_quality, fitted_spectrum
        
    except Exception as e:
        print(f"\n✗ Ошибка при фиттинге резонатора: {e}")
        # Возвращаем начальные параметры
        fit_quality = {'r_squared': 0.0, 'rmse': float('inf')}
        fitted_spectrum = None
        return initial_params, fit_quality, fitted_spectrum


# =============================================================================
# АВТОМАТИЧЕСКАЯ КАЛИБРОВКА И ОЦЕНКА ПАРАМЕТРОВ ИЗ ИНТЕРАКТИВНОГО РЕЖИМА
# =============================================================================

def find_dip_around_point(data, freq_click, field_click, search_radius_freq=0.05, search_radius_field=10):
    """
    Найти яркий провал (минимум) в окрестности точки клика
    
    Ищет минимум амплитуды S-параметра в прямоугольной окрестности точки клика.
    Используется для автоматической калибровки магнонной моды по клику.
    
    Parameters:
    -----------
    data : dict
        Данные S-параметров:
        - 'freq': массив частот (ГГц)
        - 'field': массив полей (Э)
        - 's_param': 2D массив S-параметров (комплексных или амплитуды)
    freq_click : float
        Частота клика (ГГц)
    field_click : float
        Поле клика (Э)
    search_radius_freq : float
        Радиус поиска по частоте (ГГц), по умолчанию ±0.05 ГГц = ±50 МГц
    search_radius_field : float
        Радиус поиска по полю (Э), по умолчанию ±10 Э
        
    Returns:
    --------
    calibration_point : tuple
        Калибровочная точка (field, freq) в месте минимума амплитуды
    dip_value : float
        Значение амплитуды в провале
    """
    freq = data['freq']
    field = data['field']
    s_param = data['s_param']
    
    # Определить диапазон поиска
    freq_min = freq_click - search_radius_freq
    freq_max = freq_click + search_radius_freq
    field_min = field_click - search_radius_field
    field_max = field_click + search_radius_field
    
    # Маски для фильтрации
    freq_mask = (freq >= freq_min) & (freq <= freq_max)
    field_mask = (field >= field_min) & (field <= field_max)
    
    # Проверка, что есть данные в этом диапазоне
    if not np.any(freq_mask) or not np.any(field_mask):
        print(f"  ⚠ Нет данных в окрестности клика (f={freq_click:.3f} ГГц, H={field_click:.1f} Э)")
        # Возвращаем точку клика как есть
        return (field_click, freq_click), None
    
    # Извлечь подматрицу данных
    freq_subset = freq[freq_mask]
    field_subset = field[field_mask]
    s_param_subset = np.abs(s_param)[np.ix_(field_mask, freq_mask)]
    
    # Найти минимум (провал)
    min_idx_flat = np.argmin(s_param_subset)  # Realise search of dip via scipy search peak TO DO!!!
    min_idx_2d = np.unravel_index(min_idx_flat, s_param_subset.shape)  # Realise search of dip via scipy search peak TO DO!!!
    
    field_dip = field_subset[min_idx_2d[0]]
    freq_dip = freq_subset[min_idx_2d[1]]
    dip_value = s_param_subset[min_idx_2d]
    
    print(f"  ✓ Найден провал: H={field_dip:.2f} Э, f={freq_dip:.6f} ГГц, |S|={dip_value:.6f}")
    
    return (field_dip, freq_dip), dip_value


def estimate_coherent_coupling_from_separation(interactive_params):
    """
    Оценить когерентную связь J из расстояния между модами
    
    В режиме антикроссинга расщепление мод пропорционально когерентной связи J.
    Используется расстояние между модами из интерактивного режима как начальная оценка J.
    
    Parameters:
    -----------
    interactive_params : dict
        Интерактивно выбранные параметры:
        - mode_separations: список расстояний между модами
        
    Returns:
    --------
    J_estimate : float or None
        Оценка когерентной связи (ГГц) или None, если расстояния не указаны
    """
    mode_separations = interactive_params.get('mode_separations', [])
    
    if not mode_separations:
        print("  ⚠ Расстояния между модами не выбраны в интерактивном режиме")
        return None
    
    # Извлечь расстояния по частоте
    # mode_separations - это список словарей с ключами 'freq_distance', 'field_distance', 'normalized_distance'
    distances = []
    for sep in mode_separations:
        if isinstance(sep, dict):
            # Используем расстояние по частоте как меру расщепления
            freq_dist = sep.get('freq_distance', 0)
            if freq_dist > 0:
                distances.append(freq_dist)
        else:
            # На случай, если это просто число
            if sep > 0:
                distances.append(sep)
    
    if not distances:
        print("  ⚠ Расстояния между модами по частоте не найдены или равны нулю")
        return None
    
    # Средняя оценка J как среднее расстояние
    # Физически: расщепление в антикроссинге ≈ 2*J (для сильной связи)
    # Для слабой связи: расщепление ≈ J
    # Берем консервативную оценку: J ≈ среднее_расстояние
    J_estimate = np.mean(distances)
    
    print(f"  ✓ Оценка когерентной связи из расстояния между модами:")
    print(f"    Расстояния: {[f'{d:.6f}' for d in distances]} ГГц")
    print(f"    J ≈ {J_estimate:.6f} ГГц (среднее расстояние)")
    
    return J_estimate
