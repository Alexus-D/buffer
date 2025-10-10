"""
Функции загрузки и сохранения данных

Содержит функции для работы с файлами данных и результатов
"""

import os
from datetime import datetime

import numpy as np

import config_physics, models

# =============================================================================
# ЗАГРУЗКА ДАННЫХ
# =============================================================================

def load_s_parameter_data(filepath):
    """
    Загрузить данные S-параметра из файла
    
    Формат файла:
    - 1-я строка: частоты (ГГц)
    - 1-й столбец: магнитные поля (Э)
    - Остальные данные: S-параметры
    
    Тип S-параметра определяется из имени файла:
    - Если содержит "S21" или "s21" → тип "S21"
    - Если содержит "S12" или "s12" → тип "S12"
    - По умолчанию: "S21"
    
    Parameters:
    -----------
    filepath : str
        Путь к файлу данных
        
    Returns:
    --------
    data : dict
        Словарь с данными:
        - 'freq': массив частот (ГГц)
        - 'field': массив полей (Э)
        - 's_param': 2D массив S-параметров (комплексные числа)
        - 's_type': тип S-параметра ('S21' или 'S12')
        - 'filepath': путь к файлу
    """
    # Определение типа S-параметра из имени файла
    filename = os.path.basename(filepath).lower()
    if 's12' in filename:
        s_type = 'S12'
    else:
        s_type = 'S21'
    
    # Загрузка данных из файла
    raw_data = np.loadtxt(filepath, delimiter='\t')
    
    # Извлечение частот (первая строка, начиная со 2-го элемента)
    freq = raw_data[0, 1:]
    
    # Извлечение полей (первый столбец, начиная со 2-го элемента)
    field = raw_data[1:, 0]
    
    # Извлечение S-параметров (остальная часть матрицы)
    s_param_raw = raw_data[1:, 1:]
    
    # Преобразование в комплексные числа
    # Предполагается формат: dB и амплитуда (нужно конвертировать из dB)
    s_param_magnitude = models.convert_dB_to_linear(s_param_raw)
    
    # Создание комплексного массива (пока без фазы, только амплитуда)
    s_param = s_param_magnitude.astype(complex)
    
    # Формирование словаря результата
    data = {
        'freq': freq,
        'field': field,
        's_param': s_param,
        's_type': s_type,
        'filepath': filepath
    }

    data =  filter_data_by_range(data, field_range=(config_physics.MIN_VALID_FIELD, None), freq_range=None) if config_physics.MIN_VALID_FIELD is not None else data

    return filter_data_by_range(data, field_range=config_physics.FIELD_RANGE, freq_range=config_physics.FREQ_RANGE)

# =============================================================================
# ОБРАБОТКА ДАННЫХ
# =============================================================================

def filter_data_by_range(data, field_range=None, freq_range=None):
    """
    Отфильтровать данные по диапазонам поля и частоты
    
    Parameters:
    -----------
    data : dict
        Словарь с данными
    field_range : tuple or None
        Диапазон полей (min, max)
    freq_range : tuple or None
        Диапазон частот (min, max)
        
    Returns:
    --------
    filtered_data : dict
        Отфильтрованные данные (новый словарь, исходный не изменяется)
        
    Заглушка для Этапа 1
    """
    # Создаем копию словаря, чтобы не модифицировать исходные данные
    filtered_data = {}
    
    freq_range = list(freq_range) if freq_range is not None else None
    field_range = list(field_range) if field_range is not None else None

    if freq_range is not None:
        freq_range[0] = np.min(data['freq']) if freq_range[0] is None else freq_range[0]
        freq_range[1] = np.max(data['freq']) if freq_range[1] is None else freq_range[1]
    else:
        freq_range = [np.min(data['freq']), np.max(data['freq'])]

    if field_range is not None:
        field_range[0] = np.min(data['field']) if field_range[0] is None else field_range[0]
        field_range[1] = np.max(data['field']) if field_range[1] is None else field_range[1]
    else:
        field_range = [np.min(data['field']), np.max(data['field'])]

    freq_mask = np.all([data['freq'] >= freq_range[0], data['freq'] <= freq_range[1]], axis=0) if 'freq' in data else np.ones_like(data['freq'], dtype=bool)
    field_mask = np.all([data['field'] >= field_range[0], data['field'] <= field_range[1]], axis=0) if 'field' in data else np.ones_like(data['field'], dtype=bool)
    
    # Создаем новый словарь с отфильтрованными данными
    filtered_data['freq'] = data['freq'][freq_mask]
    filtered_data['field'] = data['field'][field_mask]
    filtered_data['s_param'] = data['s_param'][np.ix_(field_mask, freq_mask)]
    
    # Копируем остальные ключи из исходного словаря
    for key in data:
        if key not in ['freq', 'field', 's_param']:
            filtered_data[key] = data[key]
    
    return filtered_data


# =============================================================================
# СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# =============================================================================

def save_fitted_parameters(params, filepath):
    """
    Сохранить подогнанные параметры в файл
    
    Parameters:
    -----------
    params : dict
        Словарь параметров
    filepath : str
        Путь к файлу для сохранения
        
    Заглушка для Этапа 1
    """
    # TODO: Реализовать сохранение параметров
    pass


def save_fitted_data(data, filepath):
    """
    Сохранить подогнанные данные в файл
    
    Parameters:
    -----------
    data : dict
        Словарь с данными и результатами подгонки
    filepath : str
        Путь к файлу для сохранения
        
    Заглушка для Этапа 1
    """
    # TODO: Реализовать сохранение данных
    pass


def save_test_data_npz(data, filepath):
    """
    Сохранить тестовые данные в формате NPZ
    
    Parameters:
    -----------
    data : dict
        Словарь с данными (freq, field, s_param, s_type)
    filepath : str
        Путь к файлу для сохранения (.npz)
    """
    np.savez(filepath,
             freq=data['freq'],
             field=data['field'],
             s_param=data['s_param'],
             s_type=data['s_type'])


def create_results_directory(base_dir='results'):
    """
    Создать директорию для результатов с timestamp
    
    Parameters:
    -----------
    base_dir : str
        Базовая директория для результатов
        
    Returns:
    --------
    results_path : str
        Путь к созданной директории
    """
    # Создание timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Полный путь к директории результатов
    results_path = os.path.join(base_dir, f'run_{timestamp}')
    
    # Создание директории, если не существует
    os.makedirs(results_path, exist_ok=True)
    
    return results_path


# =============================================================================
# ЭКСПОРТ ДАННЫХ
# =============================================================================

def export_to_csv(data, filepath):
    """
    Экспортировать данные в CSV формат
    
    Parameters:
    -----------
    data : dict
        Словарь с данными
    filepath : str
        Путь к CSV файлу
        
    Заглушка для Этапа 1
    """
    # TODO: Реализовать экспорт в CSV
    pass


def export_to_json(data, filepath):
    """
    Экспортировать данные в JSON формат
    
    Parameters:
    -----------
    data : dict
        Словарь с данными
    filepath : str
        Путь к JSON файлу
        
    Заглушка для Этапа 1
    """
    # TODO: Реализовать экспорт в JSON
    pass
