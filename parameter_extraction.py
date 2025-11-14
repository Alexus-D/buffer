"""
Модуль для извлечения параметров связанных осцилляторов из отслеженных пиков

Из собственных частот omega_+ и omega_- извлекаются параметры:
- alpha: потери магнонов
- gamma: связанные потери магнонов
- J: частота связи (coupling strength)
- Gamma: комбинированный параметр потерь sqrt(kappa*gamma)

Автор: GitHub Copilot & Alexey Kaminskiy
Дата: 2025-11-13
"""

import numpy as np
from typing import Dict, List, Tuple
import config_physics


def extract_coupling_parameters(
    peak_results: List[Dict],
    cavity_params: Dict,
    verbose: bool = True
) -> List[Dict]:
    """
    Извлекает параметры связи из результатов отслеживания пиков
    
    Из формул:
    omega_+ = (omega_c + omega_m + sqrt((omega_c-omega_m)^2 + 4*Z^2))/2
    omega_- = (omega_c + omega_m - sqrt((omega_c-omega_m)^2 + 4*Z^2))/2
    
    где Z = J - i*e^(i*theta)*Gamma
    
    Извлекаются параметры alpha, gamma, J, Gamma
    
    Parameters:
    -----------
    peak_results : list of dict
        Результаты отслеживания пиков с полями:
        - 'field': магнитное поле (Э)
        - 'f1': частота первого пика (ГГц)
        - 'w1': ширина первого пика (ГГц)
        - 'f2': частота второго пика (ГГц)
        - 'w2': ширина второго пика (ГГц)
    
    cavity_params : Dict
        Параметры резонатора из фиттинга:
        - 'wc': omega_c_real (ГГц)
        - 'kappa': потери резонатора (ГГц)
        - 'beta': дополнительные потери резонатора (ГГц)
    
    verbose : bool
        Выводить ли отладочную информацию
    
    Returns:
    --------
    coupling_results : List[Dict]
        Список словарей с извлеченными параметрами для каждого поля:
        - 'field': магнитное поле (Э)
        - 'omega_m_real': парциальная частота магнонов (ГГц)
        - 'alpha': потери магнонов (ГГц)
        - 'gamma': связанные потери магнонов (ГГц)
        - 'J': частота связи (ГГц)
        - 'Gamma': комбинированный параметр потерь (ГГц)
        - 'omega_plus_real': Re(omega_+) (ГГц)
        - 'omega_plus_imag': Im(omega_+) (ГГц)
        - 'omega_minus_real': Re(omega_-) (ГГц)
        - 'omega_minus_imag': Im(omega_-) (ГГц)
    """
    
    if verbose:
        print(f"\n{'='*70}")
        print("ИЗВЛЕЧЕНИЕ ПАРАМЕТРОВ СВЯЗИ ИЗ ОТСЛЕЖЕННЫХ ПИКОВ")
        print(f"{'='*70}")
    
    # Параметры резонатора (в ГГц)
    omega_c_real = cavity_params['wc']
    kappa = cavity_params['kappa']
    beta = cavity_params['beta']
    
    # Комплексная частота резонатора
    omega_c = omega_c_real - 1j * beta
    
    # Определяем theta (фазу) из типа S-параметра
    # theta = 0 для S21, theta = pi для S12
    s_type = config_physics.S_TYPE if hasattr(config_physics, 'S_TYPE') else 'S21'
    theta = 0 if s_type == 'S21' else np.pi
    
    if verbose:
        print(f"\nПараметры резонатора (из фиттинга):")
        print(f"  omega_c_real = {omega_c_real:.6f} ГГц")
        print(f"  kappa = {kappa:.6f} ГГц")
        print(f"  beta = {beta:.6f} ГГц")
        print(f"  omega_c = {omega_c_real:.6f} - i*{beta:.6f} ГГц")
        print(f"\nТип S-параметра: {s_type}, theta = {theta:.3f}")
        print(f"\nГиромагнитное отношение: {config_physics.GYROMAGNETIC_RATIO:.6f} ГГц/Э")
    
    coupling_results = []
    
    for i, result in enumerate(peak_results):
        field = result['field']
        
        # Собственные частоты из отслеживания пиков
        # Действительные части - это частоты пиков
        # Мнимые части - это ширины пиков (потери)
        # Ключи в результатах: 'f1', 'f2', 'w1', 'w2'
        omega_plus_real = result['f1']  # ГГц
        omega_plus_imag = -result['w1']  # ГГц (отрицательная, так как потери)
        omega_minus_real = result['f2']  # ГГц
        omega_minus_imag = -result['w2']  # ГГц
        
        omega_plus = omega_plus_real + 1j * omega_plus_imag
        omega_minus = omega_minus_real + 1j * omega_minus_imag
        
        # Парциальная частота магнонов из гиромагнитного отношения
        omega_m_real = config_physics.GYROMAGNETIC_RATIO * field  # ГГц
        
        # Из формул:
        # omega_+ + omega_- = omega_c + omega_m = omega_c + omega_m_real - i*alpha
        # Следовательно:
        sum_eigenfreqs = omega_plus + omega_minus
        
        # Комплексная частота магнонов
        omega_m = sum_eigenfreqs - omega_c
        
        # Извлекаем alpha (мнимая часть omega_m, со знаком минус)
        alpha = -omega_m.imag  # ГГц
        
        # Проверка: omega_m_real из гиромагнитного отношения должен совпадать
        omega_m_real_extracted = omega_m.real
        if verbose and i < 3:  # Показываем только для первых трех полей
            print(f"\nПоле {field:.1f} Э:")
            print(f"  omega_m_real (гиромагнитное) = {omega_m_real:.6f} ГГц")
            print(f"  omega_m_real (из суммы) = {omega_m_real_extracted:.6f} ГГц")
            print(f"  Разница: {abs(omega_m_real - omega_m_real_extracted):.6e} ГГц")
        
        # Используем omega_m_real из гиромагнитного отношения (более точно)
        omega_m = omega_m_real - 1j * alpha
        
        # Из формул:
        # omega_+ - omega_- = sqrt((omega_c - omega_m)^2 + 4*Z^2)
        # где Z = J - i*e^(i*theta)*Gamma
        
        diff_eigenfreqs = omega_plus - omega_minus
        Delta = omega_c - omega_m
        
        # Вычисляем Z^2
        # (omega_+ - omega_-)^2 = Delta^2 + 4*Z^2
        Z_squared = (diff_eigenfreqs**2 - Delta**2) / 4
        
        # Z = J - i*e^(i*theta)*Gamma
        # Z^2 = (J - i*e^(i*theta)*Gamma)^2
        # Z^2 = J^2 - 2*i*J*e^(i*theta)*Gamma - Gamma^2
        
        # Для theta = 0: Z = J - i*Gamma
        # Z^2 = J^2 - 2*i*J*Gamma - Gamma^2 = (J^2 - Gamma^2) - 2*i*J*Gamma
        
        # Для theta = pi: Z = J + i*Gamma
        # Z^2 = J^2 + 2*i*J*Gamma - Gamma^2 = (J^2 - Gamma^2) + 2*i*J*Gamma
        
        # Извлекаем J и Gamma из Z^2
        # Re(Z^2) = J^2 - Gamma^2
        # Im(Z^2) = ±2*J*Gamma (+ для theta=pi, - для theta=0)
        
        Z_squared_real = Z_squared.real
        Z_squared_imag = Z_squared.imag
        
        # Решаем систему:
        # J^2 - Gamma^2 = Z_squared_real
        # ±2*J*Gamma = Z_squared_imag
        
        sign = 1 if theta == np.pi else -1
        
        # Из второго уравнения: Gamma = sign * Z_squared_imag / (2*J)
        # Подставляем в первое: J^2 - (sign * Z_squared_imag / (2*J))^2 = Z_squared_real
        # J^2 - Z_squared_imag^2 / (4*J^2) = Z_squared_real
        # 4*J^4 - 4*J^2*Z_squared_real - Z_squared_imag^2 = 0
        
        # Квадратное уравнение относительно J^2:
        # a = 4, b = -4*Z_squared_real, c = -Z_squared_imag^2
        a = 4
        b = -4 * Z_squared_real
        c = -Z_squared_imag**2
        
        discriminant = b**2 - 4*a*c
        
        if discriminant < 0:
            if verbose:
                print(f"  ⚠ Отрицательный дискриминант для поля {field:.1f} Э")
            J = np.nan
            Gamma = np.nan
            gamma = np.nan
        else:
            J_squared = (-b + np.sqrt(discriminant)) / (2*a)  # Берем положительный корень
            
            if J_squared < 0:
                if verbose:
                    print(f"  ⚠ Отрицательное J^2 для поля {field:.1f} Э")
                J = np.nan
                Gamma = np.nan
                gamma = np.nan
            else:
                J = np.sqrt(J_squared)  # ГГц
                
                # Вычисляем Gamma
                if abs(J) < 1e-10:
                    Gamma = 0
                    gamma = 0
                else:
                    Gamma = sign * Z_squared_imag / (2 * J)  # ГГц
                    
                    # Вычисляем gamma из соотношения Gamma = sqrt(kappa*gamma)
                    # gamma = Gamma^2 / kappa
                    if Gamma < 0:
                        if verbose:
                            print(f"  ⚠ Отрицательное Gamma={Gamma:.6f} для поля {field:.1f} Э")
                        # Берем модуль для вычисления gamma
                        gamma = Gamma**2 / kappa  # ГГц (всегда положительная)
                        # И сохраняем Gamma как положительное
                        Gamma = abs(Gamma)
                    else:
                        gamma = Gamma**2 / kappa  # ГГц
                
                if verbose and i < 5:
                    print(f"  alpha = {alpha:.6f} ГГц")
                    print(f"  J = {J:.6f} ГГц")
                    print(f"  Gamma = {Gamma:.6f} ГГц")
                    print(f"  gamma = {gamma:.6f} ГГц")
        
        # Сохраняем результаты
        coupling_results.append({
            'field': field,
            'omega_m_real': omega_m_real,
            'alpha': alpha,
            'gamma': gamma,
            'J': J,
            'Gamma': Gamma,
            'omega_plus_real': omega_plus_real,
            'omega_plus_imag': omega_plus_imag,
            'omega_minus_real': omega_minus_real,
            'omega_minus_imag': omega_minus_imag
        })
    
    if verbose:
        print(f"\n✓ Извлечено параметров для {len(coupling_results)} значений поля")
        print(f"{'='*70}\n")
    
    return coupling_results


def compute_s_parameter_model(
    freq: np.ndarray,
    field: np.ndarray,
    cavity_params: Dict,
    coupling_results: List[Dict],
    s_type: str = 'S21'
) -> np.ndarray:
    """
    Вычисляет S-параметр по модели связанных осцилляторов
    
    Использует формулу:
    S(omega) = 1 + kappa / (i*(omega-omega_c_real) - (kappa+beta) 
               + (-(i*J+Gamma*e^(i*theta))^2)/(i*(omega-omega_m_real) - (alpha-gamma)))
    
    Parameters:
    -----------
    freq : np.ndarray
        Массив частот (ГГц)
    field : np.ndarray
        Массив магнитных полей (Э)
    cavity_params : Dict
        Параметры резонатора
    coupling_results : List[Dict]
        Извлеченные параметры связи для каждого поля
    s_type : str
        Тип S-параметра ('S21' или 'S12')
    
    Returns:
    --------
    s_param : np.ndarray
        Комплексный массив S-параметров размером (len(field), len(freq))
    """
    
    omega_c_real = cavity_params['wc']
    kappa = cavity_params['kappa']
    beta = cavity_params['beta']
    
    # theta определяет фазу
    theta = 0 if s_type == 'S21' else np.pi
    
    # Инициализируем массив для S-параметра
    s_param = np.zeros((len(field), len(freq)), dtype=complex)
    
    # Вычисляем для каждого значения поля
    for i, field_val in enumerate(field):
        # Находим ближайший результат coupling_results
        idx = np.argmin(np.abs(np.array([r['field'] for r in coupling_results]) - field_val))
        params = coupling_results[idx]
        
        omega_m_real = params['omega_m_real']
        alpha = params['alpha']
        gamma = params['gamma']
        J = params['J']
        Gamma = params['Gamma']
        
        # Пропускаем, если параметры некорректны
        if np.isnan(J) or np.isnan(Gamma) or np.isnan(alpha) or np.isnan(gamma):
            s_param[i, :] = np.nan
            continue
        
        # Вычисляем S(omega) для всех частот
        for j, f in enumerate(freq):
            omega = f  # ГГц (угловая частота в единицах 2*pi*ГГц)
            
            # Знаменатель основной дроби
            denom_main = 1j * (omega - omega_c_real) - (kappa + beta)
            
            # Числитель второго слагаемого в знаменателе
            Z = -1j * J - Gamma * np.exp(1j * theta)
            numerator_correction = -(Z**2)
            
            # Знаменатель второго слагаемого
            denom_correction = 1j * (omega - omega_m_real) - (alpha - gamma)
            
            # Полный знаменатель
            full_denom = denom_main + numerator_correction / denom_correction
            
            # S-параметр
            s_param[i, j] = 1 + kappa / full_denom
    
    return s_param
