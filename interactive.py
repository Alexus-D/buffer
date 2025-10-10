"""
Модуль интерактивного взаимодействия с контурными картами

Позволяет получать начальные параметры для фиттинга путем кликов мыши на контурной карте
"""

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np


class InteractiveParameterSelector:
    """
    Класс для интерактивного выбора параметров на контурной карте
    
    Позволяет пользователю выбирать:
    - Калибровку магнонной моды (поле, частота)
    - Частоту резонатора
    - Ширину магнонной моды
    - Ширину моды резонатора
    - Расстояние между модами (для антикроссинга)
    """
    
    def __init__(self, data, fig, ax):
        """
        Инициализация селектора параметров
        
        Parameters:
        -----------
        data : dict
            Данные контурной карты (freq, field, s_param)
        fig : matplotlib.figure.Figure
            Фигура matplotlib
        ax : matplotlib.axes.Axes
            Оси графика с контурной картой
        """
        self.data = data
        self.fig = fig
        self.ax = ax
        
        # Текущий режим выбора
        self.mode = None
        
        # Сохраненные параметры
        self.magnon_calibrations = []  # [(field, freq), ...]
        self.cavity_frequency = None
        self.magnon_widths = []  # [(freq_min, freq_max), ...]
        self.cavity_width = None  # (freq_min, freq_max)
        self.mode_separations = []  # [distance, ...]
        self.cavity_fit_region = None  # {'freq_range': (f_min, f_max), 'field_range': (h_min, h_max)}
        
        # Временные данные для многошаговых режимов
        self.temp_points = []
        
        # Маркеры на графике
        self.markers = []
        self.lines = []
        
        # Создание кнопок
        self._create_buttons()
        
        # Подключение обработчика кликов
        self.cid_click = self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        
        # Текст с инструкциями
        self.instruction_text = self.fig.text(0.5, 0.02, '', 
                                              ha='center', va='bottom',
                                              fontsize=10, 
                                              bbox=dict(boxstyle='round', 
                                                       facecolor='wheat', 
                                                       alpha=0.8))
    
    def _create_buttons(self):
        """Создать кнопки управления"""
        # Позиции кнопок (слева от графика)
        button_width = 0.15
        button_height = 0.04
        button_left = 0.02
        button_spacing = 0.06
        
        # Стартовая позиция снизу
        button_bottom = 0.15
        
        # Кнопка: Калибровка магнонной моды
        ax_btn1 = self.fig.add_axes([button_left, button_bottom + button_spacing * 4, 
                                     button_width, button_height])
        self.btn_magnon_calib = Button(ax_btn1, 'Калибровка\nмагнона')
        self.btn_magnon_calib.on_clicked(self._mode_magnon_calibration)
        
        # Кнопка: Частота резонатора
        ax_btn2 = self.fig.add_axes([button_left, button_bottom + button_spacing * 3, 
                                     button_width, button_height])
        self.btn_cavity_freq = Button(ax_btn2, 'Частота\nрезонатора')
        self.btn_cavity_freq.on_clicked(self._mode_cavity_frequency)
        
        # Кнопка: Ширина магнонной моды
        ax_btn3 = self.fig.add_axes([button_left, button_bottom + button_spacing * 2, 
                                     button_width, button_height])
        self.btn_magnon_width = Button(ax_btn3, 'Ширина\nмагнона')
        self.btn_magnon_width.on_clicked(self._mode_magnon_width)
        
        # Кнопка: Ширина резонатора
        ax_btn4 = self.fig.add_axes([button_left, button_bottom + button_spacing * 1, 
                                     button_width, button_height])
        self.btn_cavity_width = Button(ax_btn4, 'Ширина\nрезонатора')
        self.btn_cavity_width.on_clicked(self._mode_cavity_width)
        
        # Кнопка: Расстояние между модами
        ax_btn5 = self.fig.add_axes([button_left, button_bottom, 
                                     button_width, button_height])
        self.btn_mode_separation = Button(ax_btn5, 'Расстояние\nмежду модами')
        self.btn_mode_separation.on_clicked(self._mode_separation)
        
        # Кнопка: Диапазон фита резонатора
        ax_btn6 = self.fig.add_axes([button_left, button_bottom - button_spacing * 1, 
                                     button_width, button_height])
        self.btn_cavity_fit_region = Button(ax_btn6, 'Диапазон фита\nрезонатора')
        self.btn_cavity_fit_region.on_clicked(self._mode_cavity_fit_region)
        
        # Кнопка: Сохранить
        ax_btn_save = self.fig.add_axes([button_left, 0.05, 
                                         button_width, button_height])
        self.btn_save = Button(ax_btn_save, 'Сохранить')
        self.btn_save.on_clicked(self._save_parameters)
        
        # Кнопка: Очистить
        ax_btn_clear = self.fig.add_axes([button_left + button_width + 0.01, 0.05, 
                                          button_width, button_height])
        self.btn_clear = Button(ax_btn_clear, 'Очистить')
        self.btn_clear.on_clicked(self._clear_all)
    
    def _mode_magnon_calibration(self, event):
        """Режим: выбор калибровочной точки магнона"""
        self.mode = 'magnon_calibration'
        self.temp_points = []
        self._update_instruction('Кликните на пик магнонной моды (выберите точку на контуре)')
    
    def _mode_cavity_frequency(self, event):
        """Режим: выбор частоты резонатора"""
        self.mode = 'cavity_frequency'
        self.temp_points = []
        self._update_instruction('Кликните на резонанс резонатора (выберите частоту)')
    
    def _mode_magnon_width(self, event):
        """Режим: выбор ширины магнонной моды"""
        self.mode = 'magnon_width'
        self.temp_points = []
        self._update_instruction('Кликните на левую границу пика магнона')
    
    def _mode_cavity_width(self, event):
        """Режим: выбор ширины резонатора"""
        self.mode = 'cavity_width'
        self.temp_points = []
        self._update_instruction('Кликните на левую границу пика резонатора')
    
    def _mode_separation(self, event):
        """Режим: измерение расстояния между модами"""
        self.mode = 'mode_separation'
        self.temp_points = []
        self._update_instruction('Кликните на первую моду')
    
    def _mode_cavity_fit_region(self, event):
        """Режим: выбор диапазона для фиттинга резонатора"""
        self.mode = 'cavity_fit_region'
        self.temp_points = []
        self._update_instruction('Кликните на левый нижний угол области для фита резонатора')
    
    def _on_click(self, event):
        """Обработчик клика мыши"""
        # Проверка, что клик был на основном графике
        if event.inaxes != self.ax:
            return
        
        if self.mode is None:
            return
        
        # Получение координат клика
        freq_click = event.xdata
        field_click = event.ydata
        
        if freq_click is None or field_click is None:
            return
        
        # Обработка в зависимости от режима
        if self.mode == 'magnon_calibration':
            self._handle_magnon_calibration(freq_click, field_click)
        
        elif self.mode == 'cavity_frequency':
            self._handle_cavity_frequency(freq_click, field_click)
        
        elif self.mode == 'magnon_width':
            self._handle_magnon_width(freq_click, field_click)
        
        elif self.mode == 'cavity_width':
            self._handle_cavity_width(freq_click, field_click)
        
        elif self.mode == 'mode_separation':
            self._handle_mode_separation(freq_click, field_click)
        
        elif self.mode == 'cavity_fit_region':
            self._handle_cavity_fit_region(freq_click, field_click)
    
    def _handle_magnon_calibration(self, freq, field):
        """
        Обработка выбора калибровочной точки магнона
        
        Автоматически находит яркий провал в окрестности клика и использует его как калибровку
        """
        # Автоматический поиск провала в окрестности клика
        try:
            from fitting import find_dip_around_point
            calibration_point, dip_value = find_dip_around_point(
                self.data, freq, field, 
                search_radius_freq=0.05,  # ±50 МГц
                search_radius_field=10     # ±10 Э
            )
            field_calibrated, freq_calibrated = calibration_point
        except Exception as e:
            print(f"  ⚠ Ошибка автопоиска провала: {e}")
            # Используем точку клика как есть
            field_calibrated, freq_calibrated = field, freq
        
        self.magnon_calibrations.append((field_calibrated, freq_calibrated))
        
        # Рисуем маркер в найденной точке
        marker, = self.ax.plot(freq_calibrated, field_calibrated, 'r*', markersize=15, 
                              label=f'Магнон {len(self.magnon_calibrations)}')
        self.markers.append(marker)
        
        # Добавляем текст
        text = self.ax.text(freq_calibrated, field_calibrated, f' M{len(self.magnon_calibrations)}', 
                           fontsize=10, color='red', fontweight='bold')
        self.markers.append(text)
        
        # Если автопоиск сработал, рисуем также точку клика для визуализации поиска
        if (field_calibrated != field) or (freq_calibrated != freq):
            click_marker, = self.ax.plot(freq, field, 'rx', markersize=8, alpha=0.5)
            self.markers.append(click_marker)
        
        self.fig.canvas.draw()
        
        self._update_instruction(f'Калибровка магнона добавлена: H={field_calibrated:.2f} Э, f={freq_calibrated:.6f} ГГц. '
                                f'Кликните еще раз для следующей моды или выберите другой режим.')
    
    def _handle_cavity_frequency(self, freq, field):
        """Обработка выбора частоты резонатора"""
        self.cavity_frequency = freq
        
        # Очищаем предыдущий маркер частоты резонатора
        for marker in self.markers:
            if hasattr(marker, 'get_label') and marker.get_label() == 'Резонатор':
                marker.remove()
        
        # Рисуем вертикальную линию
        line = self.ax.axvline(x=freq, color='blue', linestyle='--', linewidth=2, 
                              label='Резонатор', alpha=0.7)
        self.lines.append(line)
        
        self.fig.canvas.draw()
        
        self._update_instruction(f'Частота резонатора: {freq:.3f} ГГц')
        self.mode = None
    
    def _handle_magnon_width(self, freq, field):
        """Обработка выбора ширины магнонной моды (двухшаговый процесс)"""
        self.temp_points.append(freq)
        
        if len(self.temp_points) == 1:
            # Первый клик - левая граница
            marker, = self.ax.plot(freq, field, 'go', markersize=8)
            self.markers.append(marker)
            self.fig.canvas.draw()
            self._update_instruction('Кликните на правую границу пика магнона')
        
        elif len(self.temp_points) == 2:
            # Второй клик - правая граница
            freq_left = min(self.temp_points)
            freq_right = max(self.temp_points)
            self.magnon_widths.append((freq_left, freq_right))
            
            marker, = self.ax.plot(freq, field, 'go', markersize=8)
            self.markers.append(marker)
            
            # Рисуем прямоугольник
            field_min, field_max = self.data['field'].min(), self.data['field'].max()
            from matplotlib.patches import Rectangle
            rect = Rectangle((freq_left, field_min), freq_right - freq_left, 
                           field_max - field_min, 
                           fill=False, edgecolor='green', linewidth=2, 
                           linestyle='--', alpha=0.5)
            self.ax.add_patch(rect)
            self.markers.append(rect)
            
            self.fig.canvas.draw()
            
            width = freq_right - freq_left
            self._update_instruction(f'Ширина магнона: {width:.4f} ГГц ({freq_left:.3f} - {freq_right:.3f})')
            self.mode = None
            self.temp_points = []
    
    def _handle_cavity_width(self, freq, field):
        """Обработка выбора ширины резонатора (двухшаговый процесс)"""
        self.temp_points.append(freq)
        
        if len(self.temp_points) == 1:
            # Первый клик - левая граница
            marker, = self.ax.plot(freq, field, 'bo', markersize=8)
            self.markers.append(marker)
            self.fig.canvas.draw()
            self._update_instruction('Кликните на правую границу пика резонатора')
        
        elif len(self.temp_points) == 2:
            # Второй клик - правая граница
            freq_left = min(self.temp_points)
            freq_right = max(self.temp_points)
            self.cavity_width = (freq_left, freq_right)
            
            marker, = self.ax.plot(freq, field, 'bo', markersize=8)
            self.markers.append(marker)
            
            # Рисуем прямоугольник
            field_min, field_max = self.data['field'].min(), self.data['field'].max()
            from matplotlib.patches import Rectangle
            rect = Rectangle((freq_left, field_min), freq_right - freq_left, 
                           field_max - field_min, 
                           fill=False, edgecolor='blue', linewidth=2, 
                           linestyle='--', alpha=0.5)
            self.ax.add_patch(rect)
            self.markers.append(rect)
            
            self.fig.canvas.draw()
            
            width = freq_right - freq_left
            self._update_instruction(f'Ширина резонатора: {width:.4f} ГГц ({freq_left:.3f} - {freq_right:.3f})')
            self.mode = None
            self.temp_points = []
    
    def _handle_mode_separation(self, freq, field):
        """Обработка измерения расстояния между модами (двухшаговый процесс)"""
        self.temp_points.append((freq, field))
        
        if len(self.temp_points) == 1:
            # Первый клик - первая мода
            marker, = self.ax.plot(freq, field, 'mo', markersize=10)
            self.markers.append(marker)
            self.fig.canvas.draw()
            self._update_instruction('Кликните на вторую моду')
        
        elif len(self.temp_points) == 2:
            # Второй клик - вторая мода
            freq1, field1 = self.temp_points[0]
            freq2, field2 = self.temp_points[1]
            
            marker, = self.ax.plot(freq, field, 'mo', markersize=10)
            self.markers.append(marker)
            
            # Рисуем линию между точками
            line, = self.ax.plot([freq1, freq2], [field1, field2], 
                                'm-', linewidth=2, alpha=0.7)
            self.lines.append(line)
            
            # Вычисляем расстояние (евклидово в координатах графика)
            # Нормализуем: freq в ГГц, field в Э
            freq_norm = (freq2 - freq1) / (self.data['freq'].max() - self.data['freq'].min())
            field_norm = (field2 - field1) / (self.data['field'].max() - self.data['field'].min())
            distance = np.sqrt(freq_norm**2 + field_norm**2)
            
            # Также сохраняем абсолютные расстояния
            freq_distance = abs(freq2 - freq1)
            field_distance = abs(field2 - field1)
            
            self.mode_separations.append({
                'freq1': freq1,
                'field1': field1,
                'freq2': freq2,
                'field2': field2,
                'freq_distance': freq_distance,
                'field_distance': field_distance,
                'normalized_distance': distance
            })
            
            self.fig.canvas.draw()
            
            self._update_instruction(f'Расстояние: Δf={freq_distance:.4f} ГГц, ΔH={field_distance:.1f} Э')
            self.mode = None
            self.temp_points = []
    
    def _handle_cavity_fit_region(self, freq, field):
        """Обработка выбора диапазона для фиттинга резонатора (двухшаговый процесс)"""
        self.temp_points.append((freq, field))
        
        if len(self.temp_points) == 1:
            # Первый клик - левый нижний угол
            marker, = self.ax.plot(freq, field, 'cs', markersize=10)
            self.markers.append(marker)
            self.fig.canvas.draw()
            self._update_instruction('Кликните на правый верхний угол области для фита резонатора')
        
        elif len(self.temp_points) == 2:
            # Второй клик - правый верхний угол
            freq1, field1 = self.temp_points[0]
            freq2, field2 = self.temp_points[1]
            
            # Определяем границы области
            freq_min, freq_max = min(freq1, freq2), max(freq1, freq2)
            field_min, field_max = min(field1, field2), max(field1, field2)
            
            self.cavity_fit_region = {
                'freq_range': (float(freq_min), float(freq_max)),
                'field_range': (float(field_min), float(field_max))
            }
            
            marker, = self.ax.plot(freq, field, 'cs', markersize=10)
            self.markers.append(marker)
            
            # Рисуем прямоугольник области фиттинга
            from matplotlib.patches import Rectangle
            rect = Rectangle((freq_min, field_min), freq_max - freq_min, 
                           field_max - field_min, 
                           fill=False, edgecolor='cyan', linewidth=2, 
                           linestyle='-', alpha=0.8)
            self.ax.add_patch(rect)
            self.markers.append(rect)
            
            self.fig.canvas.draw()
            
            self._update_instruction(f'Диапазон фита резонатора: f={freq_min:.3f}-{freq_max:.3f} ГГц, '
                                    f'H={field_min:.1f}-{field_max:.1f} Э')
            self.mode = None
            self.temp_points = []
    
    def _update_instruction(self, text):
        """Обновить текст инструкции"""
        self.instruction_text.set_text(text)
        self.fig.canvas.draw()
    
    def _clear_all(self, event):
        """Очистить все маркеры и сохраненные данные"""
        # Удаляем все маркеры
        for marker in self.markers:
            marker.remove()
        for line in self.lines:
            line.remove()
        
        self.markers = []
        self.lines = []
        
        # Очищаем данные
        self.magnon_calibrations = []
        self.cavity_frequency = None
        self.magnon_widths = []
        self.cavity_width = None
        self.mode_separations = []
        self.cavity_fit_region = None
        self.temp_points = []
        self.mode = None
        
        self.fig.canvas.draw()
        self._update_instruction('Все данные очищены. Выберите режим для начала.')
    
    def _save_parameters(self, event):
        """Сохранить параметры в конфигурационный файл"""
        import config_physics
        import os
        
        # Формируем строки для сохранения
        lines = []
        lines.append("\n# =============================================================================\n")
        lines.append("# ИНТЕРАКТИВНО ВЫБРАННЫЕ ПАРАМЕТРЫ\n")
        lines.append("# =============================================================================\n")
        lines.append("\n")
        
        if self.magnon_calibrations:
            lines.append("# Калибровочные точки магнонных мод [(поле_Э, частота_ГГц), ...]\n")
            lines.append(f"INTERACTIVE_MAGNON_CALIBRATIONS = {self.magnon_calibrations}\n")
            lines.append("\n")
        
        if self.cavity_frequency is not None:
            lines.append("# Частота резонатора (ГГц)\n")
            lines.append(f"INTERACTIVE_CAVITY_FREQUENCY = {self.cavity_frequency:.6f}\n")
            lines.append("\n")
        
        if self.magnon_widths:
            lines.append("# Ширины магнонных мод [(freq_min, freq_max), ...]\n")
            lines.append(f"INTERACTIVE_MAGNON_WIDTHS = {self.magnon_widths}\n")
            lines.append("\n")
        
        if self.cavity_width is not None:
            lines.append("# Ширина резонатора (freq_min, freq_max)\n")
            lines.append(f"INTERACTIVE_CAVITY_WIDTH = {self.cavity_width}\n")
            lines.append("\n")
        
        if self.mode_separations:
            lines.append("# Расстояния между модами\n")
            lines.append(f"INTERACTIVE_MODE_SEPARATIONS = {self.mode_separations}\n")
            lines.append("\n")
        
        if self.cavity_fit_region is not None:
            lines.append("# Диапазон для фиттинга резонатора\n")
            lines.append(f"INTERACTIVE_CAVITY_FIT_REGION = {self.cavity_fit_region}\n")
            lines.append("\n")
        
        # Путь к файлу конфига
        config_path = os.path.join(os.path.dirname(config_physics.__file__), 
                                   'config_interactive.py')
        
        # Сохраняем в файл
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write('"""\n')
            f.write('Интерактивно выбранные параметры\n')
            f.write('Автоматически сгенерирован interactive.py\n')
            f.write('"""\n')
            f.writelines(lines)
        
        self._update_instruction(f'Параметры сохранены в {config_path}')
        print(f"\n{'='*70}")
        print("Параметры сохранены в config_interactive.py:")
        print(''.join(lines))
        print('='*70)
    
    def get_parameters(self):
        """
        Получить словарь с выбранными параметрами
        
        Returns:
        --------
        params : dict
            Словарь с параметрами
        """
        return {
            'magnon_calibrations': self.magnon_calibrations,
            'cavity_frequency': self.cavity_frequency,
            'magnon_widths': self.magnon_widths,
            'cavity_width': self.cavity_width,
            'mode_separations': self.mode_separations,
            'cavity_fit_region': self.cavity_fit_region
        }


def plot_interactive_contour_map(data, title=''):
    """
    Построить интерактивную контурную карту с кнопками выбора параметров
    
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
        
    Returns:
    --------
    selector : InteractiveParameterSelector
        Объект селектора параметров
    """
    import models
    
    # Извлечение данных
    freq = data['freq']
    field = data['field']
    s_param = data['s_param']
    s_type = data.get('s_type', 'S21')
    
    # Вычисление амплитуды в dB
    s_amplitude_db = models.convert_linear_to_dB(np.abs(s_param))
    
    # Создание сетки для контурной карты
    freq_grid, field_grid = np.meshgrid(freq, field)
    
    # Создание фигуры с увеличенным пространством слева для кнопок
    fig = plt.figure(figsize=(14, 8))
    
    # Основной график (смещен вправо для кнопок)
    ax = fig.add_axes([0.25, 0.15, 0.7, 0.75])
    
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
        ax.set_title(f'Интерактивный выбор параметров - {s_type}', fontsize=14)
    
    # Сетка
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Создание селектора параметров
    selector = InteractiveParameterSelector(data, fig, ax)
    
    # Начальная инструкция
    selector._update_instruction('Выберите режим, нажав на одну из кнопок слева')
    
    # Отображение
    plt.show()
    
    return selector
