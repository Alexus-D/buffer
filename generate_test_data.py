"""
Генерация тестовых данных для антикроссинга мод ФМР и резонатора

Автор: Alexey Kaminskiy
Дата создания: 2025-10-09
"""

import os

import numpy as np

import models
import config_physics

# =============================================================================
# ПАРАМЕТРЫ ГЕНЕРАЦИИ ТЕСТОВЫХ ДАННЫХ
# =============================================================================

# Диапазон частот (ГГц)
FREQ_MIN = 4.724 - 0.5
FREQ_MAX = 4.724 + 0.5
FREQ_POINTS = 1000

# Диапазон магнитных полей (Э)
FIELD_MIN = 2900 - 500
FIELD_MAX = 2900 + 500
FIELD_POINTS = 1000

# Параметры резонатора
CAVITY_PARAMS = {
    'wc': 4.7,      # Резонансная частота (ГГц)
    'kappa': 0.88,    # Внешние потери (ГГц)
    'beta': 0.015     # Внутренние потери (ГГц)
}

# Параметры связи
COUPLING_PARAMS = {
    'J': 0.0079,       # Когерентная связь (ГГц)
    # Gamma вычисляется автоматически как sqrt(kappa * gamma_magnon)
    # НЕ задавайте Gamma вручную! Она должна вычисляться в модели
    # 'Gamma': sqrt(0.003 * 0.004) = 0.003464
}

# Параметры магнонных мод
MAGNON_PARAMS = {
    'gamma': 0.000071,   # Внешние потери (ГГц)
    'alpha': 0.0011    # Внутренние потери (ГГц)
}

# Калибровочные точки для мод ФМР [поле (Э), частота (ГГц)]
MODE_CALIBRATIONS = [
    (2800, 4.7),     # Первая мода
    # (2900, 4.7),  # Вторая мода (раскомментировать для многомодового режима)
]

# Соотношение сигнал/шум (SNR в dB)
# Чем больше значение, тем меньше шум
SIGNAL_TO_NOISE_RATIO = 10  # dB

# Директория для сохранения тестовых данных
TEST_DATA_DIR = 'test_data'

# =============================================================================
# ФУНКЦИИ ГЕНЕРАЦИИ
# =============================================================================

def generate_anticrossing_one_mode(freq, field, cavity_params, coupling_params, 
                                   magnon_params, mode_calibration, snr_db=40):
    """
    Генерация тестовых данных антикроссинга для одной моды
    
    Parameters:
    -----------
    freq : array-like
        Массив частот (ГГц)
    field : array-like
        Массив магнитных полей (Э)
    cavity_params : dict
        Параметры резонатора (wc, kappa, beta)
    coupling_params : dict
        Параметры связи (J, Gamma)
    magnon_params : dict
        Параметры магнонов (gamma, alpha)
    mode_calibration : tuple
        Калибровочная точка (H0, w0)
    snr_db : float
        Соотношение сигнал/шум в dB
        
    Returns:
    --------
    s_param : ndarray
        2D массив комплексных S-параметров [field x freq]
    """
    # Создание сетки
    freq_grid, field_grid = np.meshgrid(freq, field)
    
    # Подготовка параметров для модели
    H0, w0 = mode_calibration
    params = {
        **cavity_params,
        'J': coupling_params['J'],  # Только J, Gamma вычислится автоматически
        **magnon_params,
        'gamma_g': config_physics.GYROMAGNETIC_RATIO,
        'H0': H0,
        'w0': w0
    }
    # Gamma не передаем - она вычислится внутри модели как sqrt(kappa * gamma)
    
    # Генерация чистого сигнала
    s_param_clean = models.anticrossing_one_mode_model(freq_grid, field_grid, params)
    
    # Добавление шума
    s_param_noisy = add_noise(s_param_clean, snr_db)
    
    return np.abs(s_param_noisy)


def generate_anticrossing_multimode(freq, field, cavity_params, coupling_params,
                                   magnon_params, mode_calibrations, snr_db=40):
    """
    Генерация тестовых данных антикроссинга для нескольких мод
    
    Parameters:
    -----------
    freq : array-like
        Массив частот (ГГц)
    field : array-like
        Массив магнитных полей (Э)
    cavity_params : dict
        Параметры резонатора (wc, kappa, beta)
    coupling_params : dict
        Параметры связи (J, Gamma)
    magnon_params : dict
        Параметры магнонов (gamma, alpha)
    mode_calibrations : list of tuples
        Список калибровочных точек [(H0_1, w0_1), (H0_2, w0_2), ...]
    snr_db : float
        Соотношение сигнал/шум в dB
        
    Returns:
    --------
    s_param : ndarray
        2D массив комплексных S-параметров [field x freq]
    """
    # Создание сетки
    freq_grid, field_grid = np.meshgrid(freq, field)
    
    # Подготовка параметров для модели
    num_modes = len(mode_calibrations)
    params = {
        **cavity_params,
        'J': coupling_params['J'],  # Только J, Gamma вычислится автоматически
        **magnon_params,
        'gamma_g': config_physics.GYROMAGNETIC_RATIO,
        'mode_calibrations': mode_calibrations
    }
    # Gamma не передаем - она вычислится внутри модели как sqrt(kappa * gamma)
    
    # Генерация чистого сигнала
    s_param_clean = models.anticrossing_multimode_model(freq_grid, field_grid, 
                                                        params, num_modes)
    
    # Добавление шума
    s_param_noisy = add_noise(s_param_clean, snr_db)
    
    return np.abs(s_param_noisy)


def add_noise(s_param, snr_db):
    """
    Добавление гауссовского шума к S-параметрам
    
    Parameters:
    -----------
    s_param : ndarray
        Массив комплексных S-параметров
    snr_db : float
        Соотношение сигнал/шум в dB
        
    Returns:
    --------
    s_param_noisy : ndarray
        S-параметры с добавленным шумом
    """
    # Расчет мощности сигнала
    signal_power = np.mean(np.abs(s_param)**2)
    
    # Расчет мощности шума из SNR
    snr_linear = 10**(snr_db / 10)
    noise_power = signal_power / snr_linear
    noise_std = np.sqrt(noise_power / 2)  # Делим на 2 для real и imag
    
    # Генерация комплексного шума
    noise_real = np.random.normal(0, noise_std, s_param.shape)
    noise_imag = np.random.normal(0, noise_std, s_param.shape)
    noise = noise_real + 1j * noise_imag
    
    return s_param + noise


def save_test_data(freq, field, s_param, filename):
    """
    Сохранение тестовых данных в файл
    
    Формат файла:
    - 1-я строка: частоты
    - 1-й столбец: поля
    - Остальное: амплитуда S-параметров в dB
    
    Parameters:
    -----------
    freq : array-like
        Массив частот (ГГц)
    field : array-like
        Массив магнитных полей (Э)
    s_param : ndarray
        2D массив комплексных S-параметров
    filename : str
        Имя файла для сохранения
    """
    # Создание директории, если не существует
    os.makedirs(TEST_DATA_DIR, exist_ok=True)
    
    # Преобразование амплитуды в dB
    s_amplitude_db = models.convert_linear_to_dB(np.abs(s_param))
    
    # Создание матрицы для сохранения
    nfields = len(field)
    nfreqs = len(freq)
    data_matrix = np.zeros((nfields + 1, nfreqs + 1))
    
    # Заполнение первой строки (частоты, с пустым элементом [0,0])
    data_matrix[0, 1:] = freq
    
    # Заполнение первого столбца (поля)
    data_matrix[1:, 0] = field
    
    # Заполнение S-параметров
    data_matrix[1:, 1:] = s_amplitude_db
    
    # Сохранение в файл
    filepath = os.path.join(TEST_DATA_DIR, filename)
    np.savetxt(filepath, data_matrix, delimiter='\t', fmt='%.6f')
    
    print(f"  Тестовые данные сохранены: {filepath}")


# =============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# =============================================================================

def main():
    """
    Генерация тестовых данных
    """
    print("=" * 70)
    print("Генерация тестовых данных антикроссинга")
    print("=" * 70)
    
    # Создание массивов частот и полей
    freq = np.linspace(FREQ_MIN, FREQ_MAX, FREQ_POINTS)
    field = np.linspace(FIELD_MIN, FIELD_MAX, FIELD_POINTS)
    
    print(f"\nПараметры генерации:")
    print(f"  Частоты: {FREQ_MIN} - {FREQ_MAX} ГГц ({FREQ_POINTS} точек)")
    print(f"  Поля: {FIELD_MIN} - {FIELD_MAX} Э ({FIELD_POINTS} точек)")
    print(f"  Количество мод: {len(MODE_CALIBRATIONS)}")
    print(f"  SNR: {SIGNAL_TO_NOISE_RATIO} dB")
    
    # Генерация данных
    if len(MODE_CALIBRATIONS) == 1:
        print(f"\nГенерация данных для одной моды...")
        s_param = generate_anticrossing_one_mode(
            freq, field,
            CAVITY_PARAMS,
            COUPLING_PARAMS,
            MAGNON_PARAMS,
            MODE_CALIBRATIONS[0],
            SIGNAL_TO_NOISE_RATIO
        )
        filename = 'TestData_OneMode_S21.txt'
    else:
        print(f"\nГенерация данных для {len(MODE_CALIBRATIONS)} мод...")
        s_param = generate_anticrossing_multimode(
            freq, field,
            CAVITY_PARAMS,
            COUPLING_PARAMS,
            MAGNON_PARAMS,
            MODE_CALIBRATIONS,
            SIGNAL_TO_NOISE_RATIO
        )
        filename = f'TestData_{len(MODE_CALIBRATIONS)}Modes_S21.txt'

    # NOTE: НЕ конвертируем в dB здесь - это делается внутри save_test_data!
    # s_param уже амплитуда (из generate_anticrossing_one_mode)

    print(f"✓ Данные сгенерированы")
    print(f"  Размер массива: {s_param.shape}")
    
    # Сохранение данных
    print(f"\nСохранение данных...")
    save_test_data(freq, field, s_param, filename)
    
    print("\n" + "=" * 70)
    print("Генерация завершена")
    print("=" * 70)


if __name__ == "__main__":
    main()
