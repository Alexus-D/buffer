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
        self.selected_peaks = []  # [(field, freq), (field, freq)] - для отслеживания пиков
        self.example_fields = []  # [field1, field2, ...] - поля для примеров фиттинга
        
        # Временные данные для многошаговых режимов
        self.temp_points = []
        
        # Маркеры на графике
        self.markers = []
        self.lines = []
        
        # Создание кнопок
        self.buttons_params = [['Калибровка\nмагнона', 'magnon_calibration', 'Кликните на пик магнонной моды (выберите точку на контуре)'],
                               ['Частота\nрезонатора', 'cavity_frequency', 'Кликните на пик резонатора (выберите точку на контуре)'],
                               ['Ширина\nмагнона', 'magnon_width', 'Выберите диапазон частот для магнонной моды'],
                               ['Ширина\nрезонатора', 'cavity_width', 'Выберите диапазон частот для резонатора'],
                               ['Расстояние\nмежду модами', 'separation', 'Выберите расстояние между модами'],
                               ['Диапазон фита\nрезонатора', 'cavity_fit_region', 'Выберите диапазон фита для резонатора'],
                               ['Выбрать\n2 пика', 'select_peaks', 'Выберите два пика на контурной карте'],
                               ['Примеры\nфиттинга', 'example_fields', 'Выберите поля для примеров фиттинга']]
        
        self.save_button_params = ['Сохранить', self._save_parameters]
        self.clear_button_params = ['Очистить', self._clear_all]
        self.buttons = {}
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

        for label, mode, instruction in self.buttons_params:
            ax_btn = self.fig.add_axes([button_left, button_bottom, 
                                        button_width, button_height])
            self.buttons[label] = Button(ax_btn, label)
            self.buttons[label].on_clicked(lambda event, m=mode, i=instruction: self._change_mode(m, i))
            button_bottom += button_spacing
        
        # Кнопка: Сохранить
        ax_btn_save = self.fig.add_axes([button_left, 0.05, 
                                         button_width, button_height])
        self.buttons['Сохранить'] = Button(ax_btn_save, self.save_button_params[0])
        self.buttons['Сохранить'].on_clicked(self.save_button_params[1])
        
        # Кнопка: Очистить
        ax_btn_clear = self.fig.add_axes([button_left + button_width + 0.01, 0.05, 
                                          button_width, button_height])
        self.buttons['Очистить'] = Button(ax_btn_clear, self.clear_button_params[0])
        self.buttons['Очистить'].on_clicked(self.clear_button_params[1])
    
    def _change_mode(self, new_mode, instruction):
        """Изменить режим выбора и обновить инструкцию"""
        self.mode = new_mode
        self.temp_points = []
        if new_mode == 'select_peaks':
            self.selected_peaks = []
        self._update_instruction(instruction)
    
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
        
        elif self.mode == 'select_peaks':
            self._handle_select_peaks(freq_click, field_click)
        
        elif self.mode == 'example_fields':
            self._handle_example_fields(freq_click, field_click)
    
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
    
    def _handle_select_peaks(self, freq, field):
        """
        Обработка выбора двух пиков для отслеживания
        
        Автоматически находит ближайший пик в окрестности клика
        """
        # Автоматический поиск пика в окрестности клика
        try:
            import peak_tracking
            peak_location, peak_value = peak_tracking.find_nearest_peak(
                self.data, freq, field,
                search_radius_freq=0.005  # ±5 МГц
            )
            field_peak, freq_peak = peak_location
        except Exception as e:
            print(f"  ⚠ Ошибка автопоиска пика: {e}")
            # Используем точку клика как есть
            field_peak, freq_peak = field, freq
        
        self.selected_peaks.append((field_peak, freq_peak))
        
        # Цвета для пиков
        colors = ['orange', 'purple']
        color = colors[len(self.selected_peaks) - 1]
        
        # Рисуем маркер в найденной точке
        marker, = self.ax.plot(freq_peak, field_peak, '*', color=color, 
                              markersize=20, markeredgecolor='black', markeredgewidth=1.5,
                              label=f'Пик {len(self.selected_peaks)}')
        self.markers.append(marker)
        
        # Добавляем текст
        text = self.ax.text(freq_peak, field_peak, f' P{len(self.selected_peaks)}', 
                           fontsize=12, color=color, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                    edgecolor=color, alpha=0.8))
        self.markers.append(text)
        
        # Если автопоиск сработал, рисуем также точку клика для визуализации
        if (field_peak != field) or (freq_peak != freq):
            click_marker, = self.ax.plot(freq, field, 'x', color=color, 
                                        markersize=10, markeredgewidth=2, alpha=0.5)
            self.markers.append(click_marker)
        
        self.fig.canvas.draw()
        
        if len(self.selected_peaks) < 2:
            self._update_instruction(f'Пик 1 выбран: H={field_peak:.2f} Э, f={freq_peak:.6f} ГГц. '
                                    f'Кликните на второй пик')
        else:
            # Оба пика выбраны - рисуем линию между ними
            f1, h1 = self.selected_peaks[0][1], self.selected_peaks[0][0]
            f2, h2 = self.selected_peaks[1][1], self.selected_peaks[1][0]
            
            line, = self.ax.plot([f1, f2], [h1, h2], 
                                'k--', linewidth=2, alpha=0.6)
            self.lines.append(line)
            self.fig.canvas.draw()
            
            freq_diff = abs(f2 - f1)
            field_diff = abs(h2 - h1)
            self._update_instruction(f'Два пика выбраны! Пик 1: H={self.selected_peaks[0][0]:.2f} Э, '
                                    f'f={self.selected_peaks[0][1]:.6f} ГГц | '
                                    f'Пик 2: H={self.selected_peaks[1][0]:.2f} Э, '
                                    f'f={self.selected_peaks[1][1]:.6f} ГГц | '
                                    f'Разница: Δf={freq_diff:.6f} ГГц, ΔH={field_diff:.1f} Э')
            self.mode = None
    
    def _handle_example_fields(self, freq, field):
        """
        Обработка выбора полей для примеров фиттинга
        
        Пользователь кликает по полям, программа сохраняет их для визуализации примеров
        """
        import config_physics
        
        # Найти ближайшее поле из массива данных
        field_array = self.data['field']
        field_idx = np.argmin(np.abs(field_array - field))
        field_actual = field_array[field_idx]
        
        # Проверить, не выбрано ли это поле уже
        if field_actual in self.example_fields:
            print(f"  ⚠ Поле {field_actual:.2f} Э уже выбрано")
            return
        
        self.example_fields.append(field_actual)
        
        # Цвет для маркера
        color = 'red'
        
        # Рисуем горизонтальную линию на выбранном поле
        freq_range = self.data['freq']
        line, = self.ax.plot(freq_range, [field_actual]*len(freq_range), 
                            color=color, linewidth=2, alpha=0.7, linestyle='--')
        self.lines.append(line)
        
        # Добавляем текст с номером примера
        text = self.ax.text(freq_range[0], field_actual, f' Пример {len(self.example_fields)}', 
                           fontsize=10, color=color, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                    edgecolor=color, alpha=0.8))
        self.markers.append(text)
        
        self.fig.canvas.draw()
        
        num_examples = config_physics.NUM_EXAMPLE_FITS
        if len(self.example_fields) < num_examples:
            self._update_instruction(f'Поле {len(self.example_fields)}: H={field_actual:.2f} Э выбрано. '
                                    f'Кликните еще {num_examples - len(self.example_fields)} полей '
                                    f'(всего нужно {num_examples})')
        else:
            self._update_instruction(f'Выбрано {len(self.example_fields)} полей для примеров фиттинга: '
                                    f'{[f"{f:.1f}" for f in self.example_fields]} Э')
            self.mode = None
    
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
        self.selected_peaks = []  # Очистить выбранные пики
        self.example_fields = []  # Очистить поля примеров
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
        lines.append("import numpy as np\n")
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
        
        if self.selected_peaks:
            lines.append("# Выбранные пики для отслеживания [(field, freq), (field, freq)]\n")
            lines.append(f"INTERACTIVE_SELECTED_PEAKS = {self.selected_peaks}\n")
            lines.append("\n")
        
        if self.example_fields:
            lines.append("# Поля для примеров фиттинга [field1, field2, ...]\n")
            lines.append(f"INTERACTIVE_EXAMPLE_FIELDS = {self.example_fields}\n")
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
            'cavity_fit_region': self.cavity_fit_region,
            'selected_peaks': self.selected_peaks,  # Добавляем выбранные пики
            'example_fields': self.example_fields  # Добавляем поля примеров
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
    fig = plt.figure(figsize=(12, 7))
    
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
