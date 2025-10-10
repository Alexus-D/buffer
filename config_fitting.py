"""
Конфигурация параметров подгонки

Содержит границы параметров, методы подгонки и прочее, относящееся к подгонке
"""

from config_physics import (
    CAVITY_FREQUENCY,
    CAVITY_EXTERNAL_LOSS,
    CAVITY_INTERNAL_LOSS,
    COHERENT_COUPLING,
    DISSIPATIVE_COUPLING,
    MAGNON_EXTERNAL_LOSS,
    MAGNON_INTERNAL_LOSS,
)

# =============================================================================
# ПАРАМЕТРЫ ОПТИМИЗАЦИИ
# =============================================================================

# Максимальное количество итераций
MAX_ITERATIONS = 1000

# Допуск для сходимости
OPTIMIZATION_TOLERANCE = 1e-8

# Метод оптимизации ('leastsq', 'least_squares', 'minimize')
OPTIMIZATION_METHOD = 'leastsq'

# Диапазон полей для фиттинга резонатора (Э)
CAVITY_FITTING_FIELD_RANGE = (2880, 2900)

# Диапазон частот для фиттинга резонатора (ГГц)
CAVITY_FITTING_FREQ_RANGE = (3.625, 3.75)

# Процент отклонения для границ параметров (±20%)
BOUNDS_MARGIN = 0.2

# =============================================================================
# ГРАНИЦЫ ПАРАМЕТРОВ ДЛЯ РЕЗОНАТОРА
# =============================================================================
# Границы автоматически вычисляются как ±20% от начальных значений из config_physics.py

CAVITY_PARAM_BOUNDS = {
    'wc': (CAVITY_FREQUENCY * (1 - BOUNDS_MARGIN), CAVITY_FREQUENCY * (1 + BOUNDS_MARGIN)),
    'kappa': (CAVITY_EXTERNAL_LOSS * (1 - BOUNDS_MARGIN), CAVITY_EXTERNAL_LOSS * (1 + BOUNDS_MARGIN)),
    'beta': (CAVITY_INTERNAL_LOSS * (1 - BOUNDS_MARGIN), CAVITY_INTERNAL_LOSS * (1 + BOUNDS_MARGIN))
}

# =============================================================================
# ГРАНИЦЫ ПАРАМЕТРОВ ДЛЯ ФИТТИНГА АНТИКРОССИНГА
# =============================================================================
# Границы автоматически вычисляются как ±20% от начальных значений из config_physics.py

FULL_PARAM_BOUNDS = {
    'J': (COHERENT_COUPLING * (1 - BOUNDS_MARGIN), COHERENT_COUPLING * (1 + BOUNDS_MARGIN)),
    'Gamma': (DISSIPATIVE_COUPLING * (1 - BOUNDS_MARGIN), DISSIPATIVE_COUPLING * (1 + BOUNDS_MARGIN)),
    'gamma': (MAGNON_EXTERNAL_LOSS * (1 - BOUNDS_MARGIN), MAGNON_EXTERNAL_LOSS * (1 + BOUNDS_MARGIN)),
    'alpha': (MAGNON_INTERNAL_LOSS * (1 - BOUNDS_MARGIN), MAGNON_INTERNAL_LOSS * (1 + BOUNDS_MARGIN))
}

# Границы для частот магнонных мод (добавляются динамически)
FMR_FREQUENCY_BOUNDS = (4.0, 5.5)
