"""
Интерактивный ручной выбор пиков для извлечения параметров

Workflow:
1. Показать контурный график и дать пользователю выбрать значения полей кликами
2. Для каждого выбранного поля показать срез и дать пользователю:
   - Отметить положение первого пика
   - Отметить склоны первого пика для вычисления ширины
   - Отметить положение второго пика
   - Отметить склоны второго пика для вычисления ширины
3. Вычислить параметры связи из собранных данных
4. Продолжить обычную обработку (сохранение, визуализация)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import pandas as pd
from datetime import datetime
import os

import data_io
import parameter_extraction
import visualization


class ManualPeakSelector:
    """Класс для интерактивного ручного выбора пиков"""
    
    def __init__(self, freq, field, s_param):
        """
        Инициализация
        
        Parameters:
        -----------
        freq : array
            Частоты (ГГц)
        field : array
            Магнитные поля (Э)
        s_param : 2D array
            S-параметры [n_fields, n_freqs]
        """
        self.freq = freq
        self.field = field
        self.s_param = s_param
        self.s_mag = np.abs(s_param)
        
        # Выбранные значения полей
        self.selected_fields = []
        self.field_indices = []
        
        # Результаты для каждого поля
        self.results = []
        
        # Параметры резонатора (будут установлены извне)
        self.cavity_params = None
        
        # Текущее состояние
        self.current_field_idx = 0
        self.current_step = 'select_fields'  # 'select_fields' -> 'process_peaks'
        
        # Данные текущего среза
        self.current_spectrum = None
        self.plateau_level = None
        self.peak1_freq = None
        self.peak1_mag = None
        self.peak1_half_height = None
        self.peak1_slopes = []
        self.peak2_freq = None
        self.peak2_mag = None
        self.peak2_half_height = None
        self.peak2_slopes = []
        
        # Графические элементы
        self.fig = None
        self.ax = None
        self.cid_click = None
        self.info_text = None
        
    def start(self):
        """Запустить интерактивный процесс"""
        print("\n" + "=" * 70)
        print("MANUAL PEAK SELECTION MODE")
        print("=" * 70)
        print("\nStep 1: Select field values by clicking on the contour plot")
        print("        Click as many times as you want to select different fields")
        print("        Press 'Done' when finished selecting")
        
        self._show_contour_for_selection()
        
    def _show_contour_for_selection(self):
        """Показать контурный график для выбора полей"""
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        plt.subplots_adjust(bottom=0.15)
        
        # Контурный график
        FREQ_GRID, FIELD_GRID = np.meshgrid(self.freq, self.field)
        im = self.ax.contourf(FREQ_GRID, FIELD_GRID, self.s_mag, levels=50, cmap='viridis')
        self.ax.set_xlabel('Frequency (GHz)', fontsize=12)
        self.ax.set_ylabel('Field (Oe)', fontsize=12)
        self.ax.set_title('Click to select field values for analysis', fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=self.ax, label='|S|')
        
        # Информационный текст
        self.info_text = self.ax.text(0.02, 0.98, 'Selected fields: None',
                                       transform=self.ax.transAxes, fontsize=10,
                                       verticalalignment='top',
                                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Кнопка "Done"
        done_ax = plt.axes([0.8, 0.02, 0.15, 0.05])
        self.done_button = Button(done_ax, 'Done', color='lightgreen', hovercolor='0.975')
        self.done_button.on_clicked(self._on_done_field_selection)
        
        # Подключить обработчик кликов
        self.cid_click = self.fig.canvas.mpl_connect('button_press_event', self._on_click_field_selection)
        
        plt.show()
        
    def _on_click_field_selection(self, event):
        """Обработчик клика для выбора поля"""
        if event.inaxes != self.ax:
            return
            
        # Найти ближайшее значение поля
        field_value = event.ydata
        field_idx = np.argmin(np.abs(self.field - field_value))
        
        # Добавить если еще не выбрано
        if field_idx not in self.field_indices:
            self.field_indices.append(field_idx)
            self.selected_fields.append(self.field[field_idx])
            
            # Нарисовать горизонтальную линию
            self.ax.axhline(y=self.field[field_idx], color='red', linestyle='--', linewidth=2, alpha=0.7)
            
            # Обновить текст
            fields_str = ', '.join([f'{f:.1f}' for f in sorted(self.selected_fields)])
            self.info_text.set_text(f'Selected fields ({len(self.selected_fields)}): {fields_str} Oe')
            
            self.fig.canvas.draw()
            
            print(f"Selected field: {self.field[field_idx]:.1f} Oe")
            
    def _on_done_field_selection(self, event):
        """Завершить выбор полей и начать обработку пиков"""
        if len(self.selected_fields) == 0:
            print("\nNo fields selected! Please click on the plot to select at least one field.")
            return
            
        print(f"\nField selection complete. Selected {len(self.selected_fields)} fields.")
        print("Starting peak analysis...")
        
        # Отключить обработчик кликов
        self.fig.canvas.mpl_disconnect(self.cid_click)
        plt.close(self.fig)
        
        # Сортировать поля по возрастанию
        sorted_indices = np.argsort(self.field_indices)
        self.field_indices = [self.field_indices[i] for i in sorted_indices]
        self.selected_fields = [self.selected_fields[i] for i in sorted_indices]
        
        # Начать обработку первого поля
        self.current_step = 'process_peaks'
        self.current_field_idx = 0
        self._process_next_field()
        
    def _process_next_field(self):
        """Обработать следующее поле"""
        if self.current_field_idx >= len(self.field_indices):
            # Все поля обработаны
            self._finalize_results()
            return
            
        field_idx = self.field_indices[self.current_field_idx]
        field_value = self.field[field_idx]
        
        print(f"\n{'='*70}")
        print(f"Processing field {self.current_field_idx + 1}/{len(self.field_indices)}: {field_value:.1f} Oe")
        print(f"{'='*70}")
        
        # Получить срез
        self.current_spectrum = self.s_mag[field_idx, :]
        
        # Вычислить уровень плато (медиана)
        self.plateau_level = np.median(self.current_spectrum)
        
        # Сбросить данные пиков
        self.peak1_freq = None
        self.peak1_mag = None
        self.peak1_half_height = None
        self.peak1_slopes = []
        self.peak2_freq = None
        self.peak2_mag = None
        self.peak2_half_height = None
        self.peak2_slopes = []
        
        # Показать срез для выбора пиков
        self._show_spectrum_for_peak_selection()
        
    def _show_spectrum_for_peak_selection(self):
        """Показать спектр для выбора пиков"""
        self.fig, self.ax = plt.subplots(figsize=(14, 6))
        plt.subplots_adjust(bottom=0.2)
        
        # Конвертировать в дБ для отображения
        spectrum_db = 20 * np.log10(np.abs(self.current_spectrum))
        plateau_db = 20 * np.log10(self.plateau_level)
        
        # График спектра в дБ
        self.ax.plot(self.freq, spectrum_db, 'b-', linewidth=2, label='Spectrum')
        
        # Линия плато в дБ
        self.ax.axhline(y=plateau_db, color='gray', linestyle='--', 
                       linewidth=1.5, alpha=0.7, label=f'Plateau (median): {plateau_db:.2f} dB')
        
        self.ax.set_xlabel('Frequency (GHz)', fontsize=12)
        self.ax.set_ylabel('|S| (dB)', fontsize=12)
        self.ax.set_title(f'Field: {self.field[self.field_indices[self.current_field_idx]]:.1f} Oe - Adjust view then click Ready',
                         fontsize=14, fontweight='bold')
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()
        
        # Информационный текст
        self.info_text = self.ax.text(0.02, 0.98, 'PAUSED: Adjust zoom/view, then click Ready',
                                       transform=self.ax.transAxes, fontsize=11,
                                       verticalalignment='top', fontweight='bold',
                                       bbox=dict(boxstyle='round', facecolor='orange', alpha=0.8))
        
        # Кнопки управления
        ready_ax = plt.axes([0.75, 0.02, 0.1, 0.05])
        self.ready_button = Button(ready_ax, 'Ready', color='lightgreen', hovercolor='0.975')
        self.ready_button.on_clicked(self._on_ready_to_click)
        
        pause_ax = plt.axes([0.86, 0.02, 0.1, 0.05])
        self.pause_button = Button(pause_ax, 'Pause', color='lightyellow', hovercolor='0.975')
        self.pause_button.on_clicked(self._on_pause_clicking)
        
        # Флаг паузы (изначально на паузе)
        self.is_paused = True
        
        # Обработчик кликов (изначально НЕ подключен)
        self.cid_click = None
        
        plt.show()
        
    def _on_ready_to_click(self, event):
        """Обработчик кнопки Ready - разрешить клики"""
        self.is_paused = False
        self.info_text.set_text('Step: Click on Peak 1 maximum')
        self.info_text.set_bbox(dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        
        # Подключить обработчик кликов
        if self.cid_click is None:
            self.cid_click = self.fig.canvas.mpl_connect('button_press_event', self._on_click_peak_selection)
        
        self.fig.canvas.draw()
        print("  Ready to click. Start marking peaks.")
        
    def _on_pause_clicking(self, event):
        """Обработчик кнопки Pause - приостановить клики для приближения"""
        self.is_paused = True
        self.info_text.set_text('PAUSED: Adjust zoom/view, then click Ready')
        self.info_text.set_bbox(dict(boxstyle='round', facecolor='orange', alpha=0.8))
        self.fig.canvas.draw()
        print("  Paused. You can zoom/pan without clicking peaks. Click Ready when done.")
    
    def _on_click_peak_selection(self, event):
        """Обработчик клика для выбора пиков"""
        if event.inaxes != self.ax:
            return
        
        # Игнорировать клики в режиме паузы
        if self.is_paused:
            return
            
        click_freq = event.xdata
        click_mag_db = event.ydata
        
        # Конвертировать из дБ обратно в линейный масштаб
        click_mag = 10**(click_mag_db / 20)
        
        # Определить текущий шаг
        if self.peak1_freq is None:
            # Клик на первый пик
            self._process_peak1_maximum(click_freq, click_mag)
        elif len(self.peak1_slopes) < 2:
            # Клик на склоны первого пика
            self._process_peak1_slope(click_freq)
        elif self.peak2_freq is None:
            # Клик на второй пик
            self._process_peak2_maximum(click_freq, click_mag)
        elif len(self.peak2_slopes) < 2:
            # Клик на склоны второго пика
            self._process_peak2_slope(click_freq)
        else:
            # Все данные собраны для этого поля
            pass
            
    def _process_peak1_maximum(self, click_freq, click_mag):
        """Обработать клик на максимум первого пика"""
        self.peak1_freq = click_freq
        self.peak1_mag = click_mag
        self.peak1_half_height = (self.plateau_level + click_mag) / 2
        
        # Конвертировать в дБ для отображения
        click_mag_db = 20 * np.log10(click_mag)
        half_height_db = 20 * np.log10(self.peak1_half_height)
        
        # Нарисовать маркер пика
        self.ax.plot(click_freq, click_mag_db, 'ro', markersize=10, label='Peak 1')
        
        # Нарисовать линию полувысоты
        self.ax.axhline(y=half_height_db, color='red', linestyle=':', 
                       linewidth=1.5, alpha=0.7, label=f'Peak 1 half-height: {half_height_db:.2f} dB')
        
        # Обновить инструкции
        self.info_text.set_text('Step: Click on Peak 1 left slope (at half-height)')
        
        self.ax.legend()
        self.fig.canvas.draw()
        
        print(f"  Peak 1 maximum: freq={click_freq:.4f} GHz, mag={click_mag_db:.2f} dB")
        
    def _process_peak1_slope(self, click_freq):
        """Обработать клик на склон первого пика"""
        self.peak1_slopes.append(click_freq)
        
        # Нарисовать маркер
        self.ax.axvline(x=click_freq, color='red', linestyle=':', linewidth=1, alpha=0.5)
        
        if len(self.peak1_slopes) == 1:
            self.info_text.set_text('Step: Click on Peak 1 right slope (at half-height)')
            print(f"  Peak 1 left slope: freq={click_freq:.4f} GHz")
        else:
            # Вычислить ширину
            width1 = abs(self.peak1_slopes[1] - self.peak1_slopes[0])
            self.info_text.set_text(f'Peak 1 width: {width1:.4f} GHz. Now click on Peak 2 maximum')
            print(f"  Peak 1 right slope: freq={click_freq:.4f} GHz")
            print(f"  Peak 1 width: {width1:.4f} GHz")
            
        self.fig.canvas.draw()
        
    def _process_peak2_maximum(self, click_freq, click_mag):
        """Обработать клик на максимум второго пика"""
        self.peak2_freq = click_freq
        self.peak2_mag = click_mag
        self.peak2_half_height = (self.plateau_level + click_mag) / 2
        
        # Конвертировать в дБ для отображения
        click_mag_db = 20 * np.log10(click_mag)
        half_height_db = 20 * np.log10(self.peak2_half_height)
        
        # Нарисовать маркер пика
        self.ax.plot(click_freq, click_mag_db, 'go', markersize=10, label='Peak 2')
        
        # Нарисовать линию полувысоты
        self.ax.axhline(y=half_height_db, color='green', linestyle=':', 
                       linewidth=1.5, alpha=0.7, label=f'Peak 2 half-height: {half_height_db:.2f} dB')
        
        # Обновить инструкции
        self.info_text.set_text('Step: Click on Peak 2 left slope (at half-height)')
        
        self.ax.legend()
        self.fig.canvas.draw()
        
        print(f"  Peak 2 maximum: freq={click_freq:.4f} GHz, mag={click_mag_db:.2f} dB")
        
    def _process_peak2_slope(self, click_freq):
        """Обработать клик на склон второго пика"""
        self.peak2_slopes.append(click_freq)
        
        # Нарисовать маркер
        self.ax.axvline(x=click_freq, color='green', linestyle=':', linewidth=1, alpha=0.5)
        
        if len(self.peak2_slopes) == 1:
            self.info_text.set_text('Step: Click on Peak 2 right slope (at half-height)')
            print(f"  Peak 2 left slope: freq={click_freq:.4f} GHz")
        else:
            # Вычислить ширину
            width2 = abs(self.peak2_slopes[1] - self.peak2_slopes[0])
            self.info_text.set_text(f'Peak 2 width: {width2:.4f} GHz. Processing complete for this field!')
            print(f"  Peak 2 right slope: freq={click_freq:.4f} GHz")
            print(f"  Peak 2 width: {width2:.4f} GHz")
            
            # Сохранить результаты для этого поля
            self._save_current_field_results()
            
            # Подождать немного и перейти к следующему полю
            self.fig.canvas.mpl_disconnect(self.cid_click)
            plt.pause(1.0)
            plt.close(self.fig)
            
            self.current_field_idx += 1
            self._process_next_field()
            
        self.fig.canvas.draw()
        
    def _save_current_field_results(self):
        """Сохранить результаты для текущего поля"""
        field_idx = self.field_indices[self.current_field_idx]
        field_value = self.field[field_idx]
        
        width1 = abs(self.peak1_slopes[1] - self.peak1_slopes[0])
        width2 = abs(self.peak2_slopes[1] - self.peak2_slopes[0])
        
        result = {
            'field': field_value,
            'freq1': self.peak1_freq,
            'width1': width1,
            'amp1': self.peak1_mag - self.plateau_level,  # Амплитуда над плато
            'freq2': self.peak2_freq,
            'width2': width2,
            'amp2': self.peak2_mag - self.plateau_level,
        }
        
        self.results.append(result)
        
        print(f"\nResults saved for field {field_value:.1f} Oe:")
        print(f"  Peak 1: freq={result['freq1']:.4f} GHz, width={result['width1']:.4f} GHz, amp={result['amp1']:.4f}")
        print(f"  Peak 2: freq={result['freq2']:.4f} GHz, width={result['width2']:.4f} GHz, amp={result['amp2']:.4f}")
        
    def _finalize_results(self):
        """Завершить обработку и вычислить параметры связи"""
        print("\n" + "=" * 70)
        print("MANUAL PEAK SELECTION COMPLETE")
        print("=" * 70)
        print(f"\nProcessed {len(self.results)} field values")
        
        # Создать DataFrame с результатами
        df = pd.DataFrame(self.results)
        df.columns = ['Field(Э)', 'Freq1(ГГц)', 'Width1(ГГц)', 'Amp1', 'Freq2(ГГц)', 'Width2(ГГц)', 'Amp2']
        
        print("\nExtracted peak parameters:")
        print(df.to_string(index=False))
        
        # Вычислить параметры связи
        print("\n" + "=" * 70)
        print("EXTRACTING COUPLING PARAMETERS")
        print("=" * 70)
        
        if self.cavity_params is None:
            print("\nWARNING: Cavity parameters not provided!")
            print("Using placeholder values - results may be inaccurate.")
            print("Recommendation: Run cavity fitting first in main.py")
            # Заглушка для параметров резонатора
            self.cavity_params = {
                'wc': df['Freq1(ГГц)'].mean(),  # Грубая оценка
                'kappa': 0.01,
                'beta': 0.01
            }
        
        # Создать список результатов для extract_coupling_parameters
        peak_results = []
        for i, row in df.iterrows():
            peak_results.append({
                'field': row['Field(Э)'],
                'f1': row['Freq1(ГГц)'],
                'w1': row['Width1(ГГц)'],
                'f2': row['Freq2(ГГц)'],
                'w2': row['Width2(ГГц)']
            })
        
        # Использовать правильную функцию из parameter_extraction
        coupling_params = parameter_extraction.extract_coupling_parameters(
            peak_results=peak_results,
            cavity_params=self.cavity_params,
            verbose=True
        )
            
        coupling_df = pd.DataFrame(coupling_params)
        
        print("\nCoupling parameters:")
        print(coupling_df.to_string(index=False))
        
        # Сохранить результаты
        self._save_results(df, coupling_df)
        
        # Построить все графики
        self._plot_all_results(df, coupling_df)
        
        # Построить сравнение с моделью (контурные графики)
        self._plot_model_comparison(coupling_df)
        
    def _save_results(self, peak_df, coupling_df):
        """Сохранить результаты в файлы"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"results/manual_peak_selection_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Сохранить параметры пиков
        peak_file = os.path.join(output_dir, "peak_parameters.csv")
        peak_df.to_csv(peak_file, index=False)
        print(f"\nPeak parameters saved to: {peak_file}")
        
        # Сохранить параметры связи
        coupling_file = os.path.join(output_dir, "coupling_parameters.csv")
        coupling_df.to_csv(coupling_file, index=False)
        print(f"Coupling parameters saved to: {coupling_file}")
        
        self.output_dir = output_dir
        
    def _plot_all_results(self, peak_df, coupling_df):
        """Построить все графики параметров"""
        print("\nGenerating comprehensive parameter plots...")
        
        # Создать большой набор графиков (3x3)
        fig, axes = plt.subplots(3, 3, figsize=(18, 14))
        fig.suptitle('Manual Peak Selection: All Parameters vs Field', fontsize=16, fontweight='bold')
        
        fields = peak_df['Field(Э)'].values
        
        # График 1: Частоты пиков (резонансные частоты)
        axes[0, 0].plot(fields, peak_df['Freq1(ГГц)'], 'ro-', label='Peak 1 (ω₊)', markersize=6, linewidth=2)
        axes[0, 0].plot(fields, peak_df['Freq2(ГГц)'], 'bo-', label='Peak 2 (ω₋)', markersize=6, linewidth=2)
        axes[0, 0].set_xlabel('Field (Oe)', fontsize=10)
        axes[0, 0].set_ylabel('Frequency (GHz)', fontsize=10)
        axes[0, 0].set_title('Resonance Frequencies (Peaks)', fontsize=11, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # График 2: Ширины пиков
        axes[0, 1].plot(fields, peak_df['Width1(ГГц)'] * 1000, 'ro-', label='Peak 1', markersize=6, linewidth=2)
        axes[0, 1].plot(fields, peak_df['Width2(ГГц)'] * 1000, 'bo-', label='Peak 2', markersize=6, linewidth=2)
        axes[0, 1].set_xlabel('Field (Oe)', fontsize=10)
        axes[0, 1].set_ylabel('Width (MHz)', fontsize=10)
        axes[0, 1].set_title('Peak Widths', fontsize=11, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # График 3: Константа связи J
        axes[0, 2].plot(coupling_df['field'], coupling_df['J'] * 1000, 'go-', markersize=6, linewidth=2)
        axes[0, 2].set_xlabel('Field (Oe)', fontsize=10)
        axes[0, 2].set_ylabel('J (MHz)', fontsize=10)
        axes[0, 2].set_title('Coupling Constant J', fontsize=11, fontweight='bold')
        axes[0, 2].grid(True, alpha=0.3)
        
        # График 4: Gamma (комбинированные потери)
        axes[1, 0].plot(coupling_df['field'], coupling_df['Gamma'] * 1000, 'mo-', markersize=6, linewidth=2)
        axes[1, 0].set_xlabel('Field (Oe)', fontsize=10)
        axes[1, 0].set_ylabel('Γ (MHz)', fontsize=10)
        axes[1, 0].set_title('Combined Damping Γ', fontsize=11, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # График 5: gamma (магнонные потери)
        axes[1, 1].plot(coupling_df['field'], coupling_df['gamma'] * 1000, 'co-', markersize=6, linewidth=2)
        axes[1, 1].set_xlabel('Field (Oe)', fontsize=10)
        axes[1, 1].set_ylabel('γ (MHz)', fontsize=10)
        axes[1, 1].set_title('Magnon Damping γ', fontsize=11, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        # График 6: alpha (гильбертово затухание)
        axes[1, 2].plot(coupling_df['field'], coupling_df['alpha'] * 1000, 'orange', marker='o', markersize=6, linewidth=2)
        axes[1, 2].set_xlabel('Field (Oe)', fontsize=10)
        axes[1, 2].set_ylabel('α (MHz)', fontsize=10)
        axes[1, 2].set_title('Gilbert Damping α', fontsize=11, fontweight='bold')
        axes[1, 2].grid(True, alpha=0.3)
        
        # График 7: omega_c (парциальная частота резонатора)
        if self.cavity_params is not None:
            wc = self.cavity_params['wc']
            axes[2, 0].axhline(y=wc, color='red', linestyle='--', linewidth=2, label=f'ωc = {wc:.4f} GHz')
            axes[2, 0].set_xlabel('Field (Oe)', fontsize=10)
            axes[2, 0].set_ylabel('ωc (GHz)', fontsize=10)
            axes[2, 0].set_title('Cavity Frequency ωc', fontsize=11, fontweight='bold')
            axes[2, 0].legend()
            axes[2, 0].grid(True, alpha=0.3)
            axes[2, 0].set_xlim(fields.min(), fields.max())
        
        # График 8: omega_m_real (парциальная частота магнонов)
        axes[2, 1].plot(coupling_df['field'], coupling_df['omega_m_real'], 'purple', marker='o', markersize=6, linewidth=2)
        axes[2, 1].set_xlabel('Field (Oe)', fontsize=10)
        axes[2, 1].set_ylabel('ωm (GHz)', fontsize=10)
        axes[2, 1].set_title('Magnon Frequency ωm (Real Part)', fontsize=11, fontweight='bold')
        axes[2, 1].grid(True, alpha=0.3)
        
        # График 9: kappa и beta
        if self.cavity_params is not None:
            kappa = self.cavity_params['kappa']
            beta = self.cavity_params['beta']
            axes[2, 2].axhline(y=kappa * 1000, color='blue', linestyle='--', linewidth=2, label=f'κ = {kappa*1000:.2f} MHz')
            axes[2, 2].axhline(y=beta * 1000, color='green', linestyle='--', linewidth=2, label=f'β = {beta*1000:.2f} MHz')
            axes[2, 2].set_xlabel('Field (Oe)', fontsize=10)
            axes[2, 2].set_ylabel('Damping (MHz)', fontsize=10)
            axes[2, 2].set_title('Cavity Damping κ & β', fontsize=11, fontweight='bold')
            axes[2, 2].legend()
            axes[2, 2].grid(True, alpha=0.3)
            axes[2, 2].set_xlim(fields.min(), fields.max())
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        
        # Сохранить график
        plot_file = os.path.join(self.output_dir, "all_parameters.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"All parameters plot saved to: {plot_file}")
        
        plt.show()
    
    def _plot_model_comparison(self, coupling_df):
        """Построить сравнение экспериментальных данных с моделью"""
        print("\nGenerating model comparison (contour plots)...")
        
        # Восстановить модель из параметров
        print("Computing model S-parameters...")
        s_model = parameter_extraction.compute_s_parameter_model(
            freq=self.freq,
            field=self.field,
            cavity_params=self.cavity_params,
            coupling_results=coupling_df.to_dict('records'),
            s_type='S21'
        )
        
        # Вычислить модуль
        s_model_mag = np.abs(s_model)
        s_exp_mag = self.s_mag
        
        # Вычислить ошибку
        error = s_model_mag - s_exp_mag
        
        # Создать 3 контурных графика
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Manual Peak Selection: Model vs Experiment', fontsize=14, fontweight='bold')
        
        FREQ_GRID, FIELD_GRID = np.meshgrid(self.freq, self.field)
        
        # График 1: Экспериментальные данные
        im1 = axes[0].contourf(FREQ_GRID, FIELD_GRID, s_exp_mag, levels=50, cmap='viridis')
        axes[0].set_xlabel('Frequency (GHz)', fontsize=11)
        axes[0].set_ylabel('Field (Oe)', fontsize=11)
        axes[0].set_title('Experimental |S|', fontsize=12, fontweight='bold')
        plt.colorbar(im1, ax=axes[0], label='|S|')
        
        # График 2: Модель
        im2 = axes[1].contourf(FREQ_GRID, FIELD_GRID, s_model_mag, levels=50, cmap='viridis')
        axes[1].set_xlabel('Frequency (GHz)', fontsize=11)
        axes[1].set_ylabel('Field (Oe)', fontsize=11)
        axes[1].set_title('Model |S|', fontsize=12, fontweight='bold')
        plt.colorbar(im2, ax=axes[1], label='|S|')
        
        # График 3: Ошибка
        error_max = max(abs(error.min()), abs(error.max()))
        im3 = axes[2].contourf(FREQ_GRID, FIELD_GRID, error, levels=50, cmap='RdBu_r', 
                               vmin=-error_max, vmax=error_max)
        axes[2].set_xlabel('Frequency (GHz)', fontsize=11)
        axes[2].set_ylabel('Field (Oe)', fontsize=11)
        rms_error = np.sqrt(np.mean(error**2))
        axes[2].set_title(f'Error (Model - Exp), RMS={rms_error:.4f}', fontsize=12, fontweight='bold')
        plt.colorbar(im3, ax=axes[2], label='Error')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        
        # Сохранить график
        plot_file = os.path.join(self.output_dir, "model_comparison.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Model comparison plot saved to: {plot_file}")
        
        plt.show()
        
        print("\n" + "=" * 70)
        print("MANUAL PEAK SELECTION ANALYSIS COMPLETE")
        print("=" * 70)


def run_manual_peak_selection(data, cavity_params=None):
    """
    Запустить интерактивный режим ручного выбора пиков
    
    Parameters:
    -----------
    data : dict
        Словарь с данными:
        - 'freq': массив частот (ГГц)
        - 'field': массив полей (Э)
        - 's_param': комплексный массив S-параметров
    cavity_params : dict or None
        Параметры резонатора из предварительного фиттинга:
        - 'wc': резонансная частота (ГГц)
        - 'kappa': потери резонатора (ГГц)
        - 'beta': дополнительные потери (ГГц)
        Если None, будут вычислены автоматически
    """
    print("\n" + "=" * 70)
    print("MANUAL PEAK SELECTION MODE")
    print("=" * 70)
    
    freq = data['freq']
    field = data['field']
    s_param = data['s_param']
    
    print(f"\nData loaded:")
    print(f"  Frequency range: {freq[0]:.3f} - {freq[-1]:.3f} GHz ({len(freq)} points)")
    print(f"  Field range: {field[0]:.1f} - {field[-1]:.1f} Oe ({len(field)} points)")
    
    if cavity_params is not None:
        print(f"\nCavity parameters (from pre-fit):")
        print(f"  wc = {cavity_params['wc']:.6f} GHz")
        print(f"  kappa = {cavity_params['kappa']:.6f} GHz")
        print(f"  beta = {cavity_params['beta']:.6f} GHz")
    
    # Создать и запустить селектор
    selector = ManualPeakSelector(freq, field, s_param)
    selector.cavity_params = cavity_params  # Сохранить параметры резонатора
    selector.start()


if __name__ == "__main__":
    # Пример использования (не рекомендуется - лучше через main.py)
    print("\n" + "=" * 70)
    print("STANDALONE MODE (NOT RECOMMENDED)")
    print("=" * 70)
    print("\nRecommendation: Use main.py with interactive mode instead.")
    print("This will properly fit the cavity before manual peak selection.")
    print("\nContinuing with standalone mode (cavity params will be estimated)...\n")
    
    data_file = r"data\Сфера\CoherentCoupling_S21.txt"
    data = data_io.load_s_parameter_data(data_file)
    run_manual_peak_selection(data, cavity_params=None)
