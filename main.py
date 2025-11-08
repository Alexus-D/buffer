"""
Основной файл анализа данных антикроссинга мод ФМР и резонатора

Автор: Alexey Kaminskiy
Дата создания: 2025-10-09
Обновлено: 2025-10-28 - Добавлен режим отслеживания собственных частот

РЕЖИМЫ РАБОТЫ:
===============

1. ОБЫЧНЫЙ РЕЖИМ (INTERACTIVE = False):
   - Фиттинг теоретической модели связанных осцилляторов
   - Извлечение парциальных параметров (wc, wm, J, Γ, потери)
   - Использует параметры из config_physics.py

2. ИНТЕРАКТИВНЫЙ РЕЖИМ - ФИТТИНГ МОДЕЛИ (INTERACTIVE = True):
   - Выбор калибровочных точек и параметров кликами мыши
   - Предварительный фиттинг резонатора
   - Фиттинг теоретической модели с интерактивными параметрами

3. ИНТЕРАКТИВНЫЙ РЕЖИМ - ОТСЛЕЖИВАНИЕ ПИКОВ (INTERACTIVE = True + выбор 2 пиков):
   - Нажмите кнопку "Выбрать 2 пика" в интерактивном окне
   - Кликните на два пика (провала) в спектре
   - Программа автоматически отследит пики по всем полям
   - Аппроксимация двумя лоренцианами
   - Извлечение собственных частот, ширин и амплитуд пиков
   - Построение графиков зависимостей от поля
   - Сохранение результатов в CSV
"""

import os
import sys

import numpy as np

# Импорт модулей проекта
import config_physics
import config_fitting
import config_data
import models
import visualization
import data_io
import fitting
import interactive
import peak_tracking  # Новый модуль для отслеживания пиков


def main(interactive_mode=False):
    """
    Главная функция выполнения анализа
    
    Parameters:
    -----------
    interactive_mode : bool
        Если True, запускается интерактивный режим выбора параметров
    """
    print("=" * 70)
    if interactive_mode:
        print("Запуск интерактивного выбора параметров...")
    else:
        print("Запуск анализа антикроссинга мод...")
    print("=" * 70)
    
    # Загрузка реальных данных
    filepath = config_data.FILEPATH
    print(f"\nЗагрузка данных из: {filepath}")
    
    try:
        data = data_io.load_s_parameter_data(filepath)
        
        print(f"✓ Данные успешно загружены")
        print(f"  Тип S-параметра: {data['s_type']}")
        print(f"  Количество точек по частоте: {len(data['freq'])}")
        print(f"  Количество точек по полю: {len(data['field'])}")
        print(f"  Диапазон частот: {data['freq'].min():.3f} - {data['freq'].max():.3f} ГГц")
        print(f"  Диапазон полей: {data['field'].min():.1f} - {data['field'].max():.1f} Э")
        print(f"  Размер массива S-параметров: {data['s_param'].shape}")
        
    except FileNotFoundError:
        print(f"✗ ОШИБКА: Файл не найден: {filepath}")
        return
    except Exception as e:
        print(f"✗ ОШИБКА при загрузке данных: {e}")
        return
    
    # Если интерактивный режим - запускаем выбор параметров, затем фиттинг резонатора
    if interactive_mode:
        print(f"\n{'='*70}")
        print("ИНТЕРАКТИВНЫЙ РЕЖИМ")
        print(f"{'='*70}")
        print("Инструкция:")
        print("1. Нажмите на одну из кнопок слева для выбора режима")
        print("2. Кликайте мышью на контурной карте для указания параметров")
        print("3. Нажмите 'Сохранить' для сохранения параметров в config_interactive.py")
        print("4. Нажмите 'Очистить' для сброса всех выбранных параметров")
        print(f"{'='*70}\n")
        
        try:
            selector = interactive.plot_interactive_contour_map(
                data=data,
                title=f'Интерактивный выбор параметров - {data["s_type"]}'
            )
            print(f"\n{'='*70}")
            print("Интерактивный режим: выбор параметров завершен")
            print(f"{'='*70}")
            
            # Получить интерактивно выбранные параметры
            interactive_params = selector.get_parameters()
            
            # Проверить, был ли выбран диапазон фиттинга резонатора
            if interactive_params.get('cavity_fit_region') is not None:
                print(f"\n{'='*70}")
                print("ПРЕДВАРИТЕЛЬНЫЙ ФИТТИНГ РЕЗОНАТОРА")
                print(f"{'='*70}")
                
                # Создать директорию для результатов
                results_dir = data_io.create_results_directory(config_data.RESULTS_DIR)
                print(f"Директория результатов: {results_dir}")
                
                # Извлечь начальные параметры резонатора
                print("\nИзвлечение начальных параметров резонатора...")
                initial_cavity_params = fitting.estimate_cavity_parameters_from_interactive(
                    interactive_params, data, interactive_params['cavity_fit_region']
                )
                print(f"  Начальные параметры:")
                print(f"  - wc = {initial_cavity_params['wc']:.6f} ГГц")
                print(f"  - kappa = {initial_cavity_params['kappa']:.6f} ГГц")
                print(f"  - beta = {initial_cavity_params['beta']:.6f} ГГц")
                
                # Выполнить фиттинг резонатора
                print("\nВыполнение фиттинга резонатора...")
                fitted_cavity_params, fit_quality, fitted_spectrum = fitting.fit_cavity_only(
                    data, 
                    interactive_params['cavity_fit_region'],
                    initial_cavity_params
                )
                
                # Построить и сохранить график
                print("\nПостроение графика фиттинга резонатора...")
                save_path = os.path.join(results_dir, 'cavity_fit_cross_section.png')
                visualization.plot_cavity_fit_cross_section(
                    fitted_spectrum,
                    fit_quality,
                    save_path=save_path,
                    show=True
                )
                
                print(f"\n{'='*70}")
                print("Предварительный фиттинг резонатора завершен")
                print(f"Подогнанные параметры будут использованы для фиттинга антикроссинга")
                print(f"{'='*70}")
                
                # Сохраняем подогнанные параметры для использования в основном фиттинге
                fitted_cavity_params_for_main = fitted_cavity_params
                
            else:
                print("\nДиапазон фиттинга резонатора не был выбран.")
                print("Будут использованы параметры из config_physics.")
                fitted_cavity_params_for_main = None
            
            print(f"\n{'='*70}")
            print("Интерактивный режим завершен, продолжаем основной анализ...")
            print(f"{'='*70}")
            # НЕ ДЕЛАЕМ return - продолжаем выполнение!
            
            # Перезагрузить config_interactive чтобы прочитать сохраненные параметры
            import importlib
            import config_interactive
            importlib.reload(config_interactive)
            # Перезагрузить config_physics тоже, так как он импортирует config_interactive
            importlib.reload(config_physics)
            
            # Проверить, есть ли сохраненные пики в файле
            selected_peaks = getattr(config_interactive, 'INTERACTIVE_SELECTED_PEAKS', None)
            if selected_peaks:
                interactive_params['selected_peaks'] = selected_peaks
                print(f"\n✓ Загружены выбранные пики из config_interactive.py: {len(selected_peaks)} пик(а)")
            
            # Проверить, есть ли сохраненные поля для примеров
            example_fields = getattr(config_interactive, 'INTERACTIVE_EXAMPLE_FIELDS', None)
            if example_fields:
                interactive_params['example_fields'] = example_fields
                print(f"✓ Загружены поля для примеров фиттинга: {len(example_fields)} полей")
            
            # =================================================================
            # ПРОВЕРКА: БЫЛ ЛИ ВЫБРАН РЕЖИМ ОТСЛЕЖИВАНИЯ ПИКОВ
            # =================================================================
            if interactive_params.get('selected_peaks') and len(interactive_params['selected_peaks']) == 2:
                print(f"\n{'='*70}")
                print("РЕЖИМ ОТСЛЕЖИВАНИЯ СОБСТВЕННЫХ ЧАСТОТ")
                print(f"{'='*70}")
                print("Обнаружены выбранные пики для отслеживания.")
                print("Будет выполнен анализ отслеживания собственных частот системы.")
                print(f"{'='*70}\n")
                
                # Извлечь выбранные пики
                peak1, peak2 = interactive_params['selected_peaks']
                
                print(f"Выбранные пики:")
                print(f"  Пик 1: H={peak1[0]:.2f} Э, f={peak1[1]:.6f} ГГц")
                print(f"  Пик 2: H={peak2[0]:.2f} Э, f={peak2[1]:.6f} ГГц")
                
                # Создание директории для результатов (если еще не создана)
                if 'results_dir' not in locals():
                    results_dir = data_io.create_results_directory(config_data.RESULTS_DIR)
                    print(f"\nДиректория результатов: {results_dir}")
                
                # Отслеживание пиков по всем полям
                print(f"\nОтслеживание пиков по всем значениям поля...")
                print(f"ИНТЕРАКТИВНЫЙ РЕЖИМ: После каждого фиттинга проверяйте результат")
                print(f"  • Нажмите 'y' если фиттинг устраивает")
                print(f"  • Кликните на два новых пика для переподгонки")
                print(f"  • Нажмите 'n' для пропуска поля")
                print(f"  • Закройте окно для завершения")
                
                peak_results = peak_tracking.track_peaks_across_fields(
                    data, peak1, peak2, verbose=True, interactive=True
                )
                
                if peak_results:
                    # Сохранение результатов
                    print(f"\n{'='*70}")
                    print("Сохранение результатов отслеживания пиков...")
                    print(f"{'='*70}")
                    
                    peak_tracking.save_peak_tracking_results(
                        peak_results, results_dir, 'peak_parameters.csv'
                    )
                    
                    peak_tracking.save_peak_tracking_summary(
                        peak_results, results_dir, 'peak_tracking_summary.txt'
                    )
                    
                    # Визуализация результатов
                    print(f"\n{'='*70}")
                    print("Построение графиков отслеживания пиков...")
                    print(f"{'='*70}")
                    
                    # График собственных частот vs поле
                    freq_path = os.path.join(results_dir, 'eigenfrequencies_vs_field.png')
                    visualization.plot_eigenfrequencies_vs_field(
                        peak_results, save_path=freq_path, show=True
                    )
                    
                    # График ширин пиков vs поле
                    width_path = os.path.join(results_dir, 'peak_widths_vs_field.png')
                    visualization.plot_peak_widths_vs_field(
                        peak_results, save_path=width_path, show=True
                    )
                    
                    # График амплитуд пиков vs поле
                    amp_path = os.path.join(results_dir, 'peak_amplitudes_vs_field.png')
                    visualization.plot_peak_amplitudes_vs_field(
                        peak_results, save_path=amp_path, show=True
                    )
                    
                    # График качества подгонки vs поле
                    quality_path = os.path.join(results_dir, 'peak_fit_quality_vs_field.png')
                    visualization.plot_peak_fit_quality_vs_field(
                        peak_results, save_path=quality_path, show=True
                    )
                    
                    # Сводный график
                    summary_path = os.path.join(results_dir, 'peak_tracking_summary.png')
                    visualization.plot_peak_tracking_summary(
                        peak_results, save_path=summary_path, show=True
                    )
                    
                    # Примеры аппроксимации для нескольких полей
                    examples_path = os.path.join(results_dir, 'example_peak_fits.png')
                    
                    # Использовать выбранные пользователем поля или равномерное распределение
                    example_fields = interactive_params.get('example_fields')
                    if example_fields:
                        # Преобразовать значения полей в индексы
                        field_indices = []
                        for field_val in example_fields:
                            idx = np.argmin(np.abs(data['field'] - field_val))
                            field_indices.append(idx)
                        print(f"\nИспользуются выбранные поля для примеров: {len(field_indices)} полей")
                    else:
                        field_indices = None  # Функция использует равномерное распределение по умолчанию
                    
                    visualization.plot_example_peak_fits(
                        data, peak_results, save_path=examples_path, show=True, field_indices=field_indices
                    )
                    
                    print(f"\n{'='*70}")
                    print("✓ ОТСЛЕЖИВАНИЕ ПИКОВ ЗАВЕРШЕНО")
                    print(f"{'='*70}")
                    print(f"Обработано полей: {len(peak_results)}")
                    print(f"Все результаты сохранены в: {results_dir}")
                    print(f"{'='*70}\n")
                    
                    # Завершаем выполнение, так как отслеживание пиков - альтернативный режим
                    return
                else:
                    print(f"\n✗ ОШИБКА: Отслеживание пиков не удалось")
                    return  # Завершаем программу
            else:
                # В интерактивном режиме пики не были выбраны - завершаем программу
                print(f"\n{'='*70}")
                print("ЗАВЕРШЕНИЕ РАБОТЫ")
                print(f"{'='*70}")
                print("Пики для отслеживания не были выбраны.")
                print("Для фиттинга модели антикроссинга установите INTERACTIVE = False в main.py")
                print(f"{'='*70}\n")
                return
            
        except Exception as e:
            print(f"✗ ОШИБКА в интерактивном режиме: {e}")
            import traceback
            traceback.print_exc()
            return
    else:
        # Обычный режим (не интерактивный) - параметры резонатора не подгонялись
        fitted_cavity_params_for_main = None
    
    # Создание директории для результатов
    print(f"\nСоздание директории результатов...")
    results_dir = data_io.create_results_directory(config_data.RESULTS_DIR)
    print(f"✓ Директория создана: {results_dir}")
    
    # Построение и сохранение контурной карты
    print(f"\nПостроение контурной карты...")
    save_path = os.path.join(results_dir, f'contour_map_{data["s_type"]}.png')
    
    try:
        visualization.plot_contour_map(
            data=data,
            title=f'Экспериментальные данные {data["s_type"]}',
            save_path=save_path,
            show=False  # Показать график на экране
        )
        print(f"✓ Контурная карта построена и сохранена")
        
    except Exception as e:
        print(f"✗ ОШИБКА при построении графика: {e}")
        return
    
    # Фиттинг спектров для каждого поля
    print("\n" + "=" * 70)
    print("Начало фиттинга спектров...")
    print("=" * 70)
    
    # Подготовка начальных параметров
    # Проверяем, есть ли интерактивно выбранные параметры
    use_interactive_params = 'interactive_params' in locals() and interactive_params is not None
    
    # Параметры резонатора: используем подогнанные или из config_physics
    if 'fitted_cavity_params_for_main' in locals() and fitted_cavity_params_for_main is not None:
        print("\n✓ Используются подогнанные параметры резонатора из предварительного фиттинга:")
        print(f"  - wc = {fitted_cavity_params_for_main['wc']:.6f} ГГц")
        print(f"  - kappa = {fitted_cavity_params_for_main['kappa']:.6f} ГГц")
        print(f"  - beta = {fitted_cavity_params_for_main['beta']:.6f} ГГц")
        wc_initial = fitted_cavity_params_for_main['wc']
        kappa_initial = fitted_cavity_params_for_main['kappa']
        beta_initial = fitted_cavity_params_for_main['beta']
    else:
        print("\n✓ Используются параметры резонатора из config_physics")
        wc_initial = config_physics.CAVITY_FREQUENCY
        kappa_initial = config_physics.CAVITY_EXTERNAL_LOSS
        beta_initial = config_physics.CAVITY_INTERNAL_LOSS
    
    # Калибровка магнонной моды: используем интерактивную или из config_physics
    if use_interactive_params and interactive_params.get('magnon_calibrations'):
        magnon_calibrations = interactive_params['magnon_calibrations']
        print("\n✓ Используются калибровочные точки магнонов из интерактивного режима:")
        for i, (field, freq) in enumerate(magnon_calibrations, 1):
            print(f"  - Мода {i}: H0={field:.2f} Э, w0={freq:.6f} ГГц")
        # Берем первую моду для фиттинга (одномодовая модель)
        H0_initial = magnon_calibrations[0][0]
        w0_initial = magnon_calibrations[0][1]
    else:
        print("\n✓ Используются калибровочные параметры магнонов из config_physics")
        H0_initial = config_physics.FMR_MODE_CALIBRATIONS[0][0]
        w0_initial = config_physics.FMR_MODE_CALIBRATIONS[0][1]
    
    # Когерентная связь: оцениваем из расстояния между модами или берем из config_physics
    if use_interactive_params and interactive_params.get('mode_separations'):
        J_estimated = fitting.estimate_coherent_coupling_from_separation(interactive_params)
        if J_estimated is not None:
            print("\n✓ Используется оценка когерентной связи из интерактивного режима:")
            print(f"  - J ≈ {J_estimated:.6f} ГГц")
            J_initial = J_estimated
        else:
            print("\n✓ Используется когерентная связь из config_physics")
            J_initial = config_physics.COHERENT_COUPLING
    else:
        print("\n✓ Используется когерентная связь из config_physics")
        J_initial = config_physics.COHERENT_COUPLING
    
    # Собираем initial_params
    initial_params = {
        'wc': wc_initial,
        'kappa': kappa_initial,
        'beta': beta_initial,
        'J': J_initial,
        'Gamma': config_physics.DISSIPATIVE_COUPLING,
        'gamma': config_physics.MAGNON_EXTERNAL_LOSS,
        'alpha': config_physics.MAGNON_INTERNAL_LOSS,
        'H0': H0_initial,
        'w0': w0_initial,
        'gamma_g': config_physics.GYROMAGNETIC_RATIO,
        's_type': data['s_type']
    }
    
    # Границы параметров для фиттинга
    # Фиттим только: J (когерентная связь), gamma (внешние потери магнонов), alpha (внутренние потери магнонов)
    # Gamma вычисляется автоматически как sqrt(kappa * gamma)
    param_bounds = {
        'J': config_fitting.FULL_PARAM_BOUNDS['J'],
        'gamma': config_fitting.FULL_PARAM_BOUNDS['gamma'],
        'alpha': config_fitting.FULL_PARAM_BOUNDS['alpha']
    }
    
    # Выполнение фиттинга для всех спектров
    fitting_results, fitted_data = fitting.fit_all_spectra(data, initial_params, param_bounds)
    
    # Сохранение результатов фиттинга
    if fitting_results:
        results_file = os.path.join(results_dir, 'fitting_results.txt')
        print(f"\nСохранение результатов фиттинга в {results_file}...")
        
        with open(results_file, 'w', encoding='utf-8') as f:
            f.write("=" * 100 + "\n")
            f.write("РЕЗУЛЬТАТЫ ФИТТИНГА СПЕКТРОВ АНТИКРОССИНГА\n")
            f.write("=" * 100 + "\n\n")
            
            # Заголовок таблицы
            f.write(f"{'Поле (Э)':<12} {'wc (ГГц)':<12} {'kappa (ГГц)':<15} {'beta (ГГц)':<15} "
                    f"{'J (ГГц)':<12} {'Gamma (ГГц)':<15} {'gamma (ГГц)':<15} {'alpha (ГГц)':<15} "
                    f"{'R²':<10} {'RMSE':<10}\n")
            f.write("-" * 100 + "\n")
            
            # Данные
            for result in fitting_results:
                f.write(f"{result['field']:<12.2f} {result['wc']:<12.6f} {result['kappa']:<15.6f} "
                        f"{result['beta']:<15.6f} {result['J']:<12.6f} {result['Gamma']:<15.6f} "
                        f"{result['gamma']:<15.6f} {result['alpha']:<15.6f} "
                        f"{result['r_squared']:<10.6f} {result['rmse']:<10.6f}\n")
            
            # Средние значения
            f.write("\n" + "=" * 100 + "\n")
            f.write("СРЕДНИЕ ЗНАЧЕНИЯ ПАРАМЕТРОВ\n")
            f.write("=" * 100 + "\n")
            
            avg_wc = np.mean([r['wc'] for r in fitting_results])
            avg_kappa = np.mean([r['kappa'] for r in fitting_results])
            avg_beta = np.mean([r['beta'] for r in fitting_results])
            avg_J = np.mean([r['J'] for r in fitting_results])
            avg_Gamma = np.mean([r['Gamma'] for r in fitting_results])
            avg_gamma = np.mean([r['gamma'] for r in fitting_results])
            avg_alpha = np.mean([r['alpha'] for r in fitting_results])
            avg_r2 = np.mean([r['r_squared'] for r in fitting_results])
            avg_rmse = np.mean([r['rmse'] for r in fitting_results])
            
            f.write(f"\nwc (резонансная частота резонатора): {avg_wc:.6f} ГГц\n")
            f.write(f"kappa (внешние потери резонатора):   {avg_kappa:.6f} ГГц\n")
            f.write(f"beta (внутренние потери резонатора): {avg_beta:.6f} ГГц\n")
            f.write(f"J (когерентная связь):                {avg_J:.6f} ГГц\n")
            f.write(f"Gamma (диссипативная связь):          {avg_Gamma:.6f} ГГц\n")
            f.write(f"gamma (внешние потери магнонов):      {avg_gamma:.6f} ГГц\n")
            f.write(f"alpha (внутренние потери магнонов):   {avg_alpha:.6f} ГГц\n")
            f.write(f"\nСреднее R²:   {avg_r2:.6f}\n")
            f.write(f"Среднее RMSE: {avg_rmse:.6f}\n")
        
        print(f"✓ Результаты сохранены")
        
        # Вывод средних значений в консоль
        print("\n" + "=" * 70)
        print("СРЕДНИЕ ЗНАЧЕНИЯ ПАРАМЕТРОВ:")
        print("=" * 70)
        print(f"wc (резонансная частота резонатора): {avg_wc:.6f} ГГц")
        print(f"kappa (внешние потери резонатора):   {avg_kappa:.6f} ГГц")
        print(f"beta (внутренние потери резонатора): {avg_beta:.6f} ГГц")
        print(f"J (когерентная связь):                {avg_J:.6f} ГГц")
        print(f"Gamma (диссипативная связь):          {avg_Gamma:.6f} ГГц")
        print(f"gamma (внешние потери магнонов):      {avg_gamma:.6f} ГГц")
        print(f"alpha (внутренние потери магнонов):   {avg_alpha:.6f} ГГц")
        print(f"\nСреднее R²:   {avg_r2:.6f}")
        print(f"Среднее RMSE: {avg_rmse:.6f}")
    
    # Построение сравнительных контурных карт
    print("\n" + "=" * 70)
    print("Построение сравнительных контурных карт...")
    print("=" * 70)
    
    comparison_path = os.path.join(results_dir, f'comparison_contours_{data["s_type"]}.png')
    
    try:
        visualization.plot_comparison_contours(
            experimental_data=data,
            fitted_data=fitted_data,
            save_path=comparison_path,
            show=True  # Показать график на экране
        )
        print(f"✓ Сравнительные графики построены и сохранены")
        
    except Exception as e:
        print(f"✗ ОШИБКА при построении сравнительных графиков: {e}")
    
    # Построение графиков зависимости параметров от поля
    print("\n" + "=" * 70)
    print("Построение графиков параметров vs поле...")
    print("=" * 70)
    
    # Преобразование результатов в нужный формат для функций визуализации
    formatted_results = []
    for result in fitting_results:
        formatted_results.append({
            'fitted_params': {
                'field': result['field'],
                'J': result['J'],
                'gamma': result['gamma'],
                'alpha': result['alpha'],
                'Gamma': result['Gamma'],
                'wc': result['wc'],
                'kappa': result['kappa'],
                'beta': result['beta']
            },
            'fit_quality': {
                'r_squared': result['r_squared'],
                'rmse': result['rmse']
            }
        })
    
    parameters_path = os.path.join(results_dir, 'fitted_parameters_vs_field.png')
    
    try:
        visualization.plot_fitted_parameters_vs_field(
            fitting_results=formatted_results,
            save_path=parameters_path,
            show=True  # Показать график на экране
        )
        print(f"✓ Графики параметров vs поле построены и сохранены")
        
    except Exception as e:
        print(f"✗ ОШИБКА при построении графиков параметров: {e}")
    
    # Построение графиков ошибок подгонки vs поле
    print("\n" + "=" * 70)
    print("Построение графиков ошибок подгонки vs поле...")
    print("=" * 70)
    
    errors_path = os.path.join(results_dir, 'fitting_errors_vs_field.png')
    
    try:
        visualization.plot_fitting_errors_vs_field(
            fitting_results=formatted_results,
            save_path=errors_path,
            show=True  # Показать график на экране
        )
        print(f"✓ Графики ошибок подгонки построены и сохранены")
        
    except Exception as e:
        print(f"✗ ОШИБКА при построении графиков ошибок: {e}")
    
    print("\n" + "=" * 70)
    print("Анализ завершен")
    print("=" * 70)


if __name__ == "__main__":
    # РЕЖИМ: True = интерактивный выбор пиков, False = обычный фиттинг модели
    INTERACTIVE = True
    main(interactive_mode=INTERACTIVE)
