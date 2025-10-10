"""
Интерактивно выбранные параметры
Автоматически сгенерирован interactive.py
"""
import numpy as np


# =============================================================================
# ИНТЕРАКТИВНО ВЫБРАННЫЕ ПАРАМЕТРЫ
# =============================================================================

# Калибровочные точки магнонных мод [(поле_Э, частота_ГГц), ...]
INTERACTIVE_MAGNON_CALIBRATIONS = [(np.float64(2865.0), np.float64(3.410295)), (np.float64(2873.5), np.float64(3.431284)), (np.float64(2870.5), np.float64(3.423788))]

# Частота резонатора (ГГц)
INTERACTIVE_CAVITY_FREQUENCY = 3.669308

# Ширины магнонных мод [(freq_min, freq_max), ...]
INTERACTIVE_MAGNON_WIDTHS = [(np.float64(3.389634817005988), np.float64(3.391329292933854))]

# Ширина резонатора (freq_min, freq_max)
INTERACTIVE_CAVITY_WIDTH = (np.float64(3.6348864795918367), np.float64(3.7139285714285712))

# Расстояния между модами
INTERACTIVE_MODE_SEPARATIONS = [{'freq1': np.float64(3.642535714285714), 'field1': np.float64(2964.6666666666665), 'freq2': np.float64(3.680781887755102), 'field2': np.float64(2965.6666666666665), 'freq_distance': np.float64(0.038246173469387745), 'field_distance': np.float64(1.0), 'normalized_distance': np.float64(0.03859059020855642)}]

