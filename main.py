import numpy as np
import matplotlib.pyplot as plt
import pickle
import scipy as sp
import time

import config_data
import data_io
import config_physics
import useful_funcs
import click_funcs
import interactive

BUFFER_DATA_FILE = "bufer_data.pkl"

def check_prosessing_results(data, proc_func, load_results= False):
    if load_results:
        output = proc_func(data, load_results=load_results)
        return output

    while True:
        output = proc_func(data, load_results=load_results)
        params = output[-1]
        buttons = [["Confirm (double click)", ["true_func", click_funcs.true_func]],
                   ["Re-process (double click)", ["false_func", click_funcs.false_func]]]
        selector = interactive.plot_interactive_contour_map(data=output,
                                                             buttons=buttons,
                                                             title='Confirm Processed Data',
                                                             clear_buttons=False)

        confirm = selector.get_params()
        if confirm.get("true_func") is True:
            try:
                file = pickle.load(open(BUFFER_DATA_FILE, 'rb'))
            except Exception:
                file = None

            if file is not None:
                file.update(params)
            else:
                file = params
            with open(BUFFER_DATA_FILE, 'wb') as f:
                pickle.dump(file, f)
            print(f"Parameters saved to {BUFFER_DATA_FILE}.")
            break
    return output

def preprocess_data(data, load_results=False):
    if load_results:
        data_range = pickle.load(open(BUFFER_DATA_FILE, 'rb'))["preprocess"]["data_range"]
        data = data_io.filter_data_by_range(data=data,
                                            x_range=data_range['x_range'],
                                            y_range=data_range['y_range'])
        return data, {"preprocess": {"data_range": data_range}}
    
    buttons = [["Select Range", ["data_range", click_funcs.choose_range]]]
    selector = interactive.plot_interactive_contour_map(data=data,
                                                        buttons=buttons,
                                                        title='Select Data Range')

    data_range = selector.get_params()["data_range"]

    data_filtered = data_io.filter_data_by_range(data=data,
                                        x_range=data_range['x_range'],
                                        y_range=data_range['y_range'])

    return data_filtered, {"preprocess": {"data_range": data_range}}

def extract_resonator_params(data, load_results=False):
    loaded_params = pickle.load(open(BUFFER_DATA_FILE, 'rb'))['resonator_params'] if load_results else None

    if load_results:
        resonator_fit_region = loaded_params['resonator_fit_region']
        resonance_freq = loaded_params['resonance_freq']
        cavity_width = loaded_params['cavity_width']
        kappa = loaded_params['kappa']
        beta = loaded_params['beta']
        res_magnitude = loaded_params['res_magnitude']
        plato = loaded_params['plato']
    else:
        buttons = [["Select Cavity Fit Region", ["cavity_fit_region", click_funcs.choose_range]],
                ["Select Cavity Frequency", ["cavity_frequency", click_funcs.choose_point]],
                ["Select Cavity Width", ["cavity_width", lambda click, params, mode, ax: click_funcs.choose_points(click, params, mode, ax, num_points=2)]]]
        selector = interactive.plot_interactive_contour_map(data=data,
                                                            buttons=buttons,
                                                            title='Select Resonator Parameters')
        
        selected_resonator_params = selector.get_params()

        resonator_fit_region = selected_resonator_params['cavity_fit_region']
        resonance_freq = selected_resonator_params['cavity_frequency'][1]
        cavity_width = np.abs(selected_resonator_params['cavity_width'][0][1] - selected_resonator_params['cavity_width'][1][1])

    filtered_data = data_io.filter_data_by_range(data=data,
                                                 x_range=resonator_fit_region['x_range'],
                                                 y_range=resonator_fit_region['y_range'])

    freqs = filtered_data['y']
    s_value = filtered_data['z']

    s_average = np.mean(np.abs(s_value), axis=0)
    res_index = np.argmin(np.abs(freqs - resonance_freq))

    if not load_results:
        res_magnitude = s_average[res_index]
        resonance_freq = freqs[res_index]
        plato = np.max(s_average) if config_physics.PEAK_TYPE == 'minimum' else np.min(s_average)

        estimated_params = useful_funcs.estimate_cavity_params(res_magnitude,
                                                            resonance_freq,
                                                            cavity_width, plato)
        
        fitted_params = useful_funcs.fit_cavity_response(freqs, s_average, estimated_params)
        kappa = fitted_params['kappa']
        beta = fitted_params['beta']
        resonance_freq = fitted_params['resonance_freq']
        res_magnitude = fitted_params['res_magnitude']
        plato = fitted_params['plato']

    output_data = {}
    experimental_line = {'x': freqs, 'y': s_average, 'label': 'experimental average'}
    modeled_s_params = np.abs(useful_funcs.cavity_model(freqs,
                                                        kappa,
                                                        beta,
                                                        resonance_freq,
                                                        plato))
    modeled_line = {'x': freqs, 'y': modeled_s_params, 'label': 'fitted model'}
    
    experimental_line['x_label'] = 'Frequency (GHz)'
    experimental_line['y_label'] = 'S-parameter Magnitude'
    modeled_line['x_label'] = 'Frequency (GHz)'
    modeled_line['y_label'] = 'S-parameter Magnitude'

    output_data['line1'] = experimental_line
    output_data['line2'] = modeled_line
    output_data['plot_type'] = 'two_line'

    output_params = {
        'kappa': np.abs(kappa),
        'beta': np.abs(beta),
        'resonance_freq': np.abs(resonance_freq),
        'res_magnitude': np.abs(res_magnitude),
        'cavity_width': np.abs(cavity_width),
        'plato': np.abs(plato),
        'resonator_fit_region': resonator_fit_region
    }

    return output_data, {'resonator_params': output_params}

def choose_slice(data, load_results=False, num_slice=1):
    loaded_params = pickle.load(open(BUFFER_DATA_FILE, 'rb')) if load_results else None
    if load_results:
        slice_point = loaded_params[f'slice_{num_slice}_params']['slice_point']
    else:
        buttons = [["Select Slice", ["slice_point", click_funcs.choose_point]]]
        selector = interactive.plot_interactive_contour_map(data=data,
                                                            title='Select Slice',
                                                            buttons=buttons)
        
        slice_point = selector.get_params()['slice_point']
        slice_point = {'field': slice_point[0], 'freq': slice_point[1]}

    field_idx = np.argmin(np.abs(data['x'] - slice_point['field']))
    s_values = np.abs(data['z'][field_idx, :])
    freqs = data['y']

    line = {'x': freqs, 'y': s_values,
            'label': f'S-parameter at Field {slice_point["field"]}',
            'x_label': 'Frequency (GHz)',
            'y_label': 'S-parameter Magnitude',
            'field': slice_point['field']}

    slice_data = {'line': line, 'plot_type': 'one_line'}
    params = {'slice_point': slice_point}
    return slice_data, {f'slice_{num_slice}_params': params}

def choose_peaks(data, load_results=False, peak_num=1):
    if load_results:
        loaded_params = pickle.load(open('selected_params.pkl', 'rb'))[f'peak_{peak_num}_params']
        prominence = loaded_params['prominence']
        width = loaded_params['width']
        peak_freq = loaded_params['peak_freq']
    else:
        buttons = [["Select Prominence", ["prominence", lambda click, params, mode, ax: click_funcs.choose_points(click, params, mode, ax, num_points=2)]],
                    ["Select Width", ["width", lambda click, params, mode, ax: click_funcs.choose_points(click, params, mode, ax, num_points=2)]],
                    ["Peak frequency", ["peak_freq", click_funcs.choose_point]]]
        selector = interactive.plot_interactive_contour_map(data=data,
                                                            title=f'Select Prominence for Peak',
                                                            buttons=buttons)
        prominence_points = selector.get_params()['prominence']
        width_points = selector.get_params()['width']
        peak_freq = selector.get_params()['peak_freq'][0]

        prominence = np.abs(prominence_points[0][1] - prominence_points[1][1])
        width = np.abs(width_points[0][0] - width_points[1][0])

    current_peak = {'prominence': prominence,
                    'width': width,
                    'peak_freq': peak_freq}
    return None, {f"peak_{peak_num}_params": current_peak}

def magnon_calibration(data, load_results=False):
    if load_results:
        pass
    else:
        buttons = [["Select Magnon Frequencies", ["magnon_points", lambda click, params, mode, ax: click_funcs.choose_points(click, params, mode, ax, num_points=5)]]]
        selector = interactive.plot_interactive_contour_map(data=data,
                                                            title='Select Magnon Frequencies for Calibration',
                                                            buttons=buttons)
        magnon_points = selector.get_params()['magnon_points']
        fields = np.array([point[0] for point in magnon_points])
        magnon_freqs_experimental = np.array([point[1] for point in magnon_points])

        magnon_freqs_calibrated, offset = useful_funcs.calibrate_magnon_frequency(fields, magnon_freqs_experimental)

    output_params = {
        'offset': offset,
        'magnon_freqs_calibrated': magnon_freqs_calibrated
    }

    try:
        file = pickle.load(open(BUFFER_DATA_FILE, 'rb'))
    except Exception:
        file = None
    if file is not None:
        file.update({'magnon_calibration': output_params})
        with open(BUFFER_DATA_FILE, 'wb') as f:
            pickle.dump(file, f)
    else:
        with open(BUFFER_DATA_FILE, 'wb') as f:
            pickle.dump({'magnon_calibration': output_params}, f)

    return None, {'magnon_calibration': output_params}

def find_peaks(data, load_results=False):
    slice_1, _ = check_prosessing_results(data,
                                          lambda data, load_results: choose_slice(data,
                                                                                  load_results=load_results,
                                                                                  num_slice=1),
                                          load_results=load_results)
    slice_2, _ = check_prosessing_results(data,
                                          lambda data, load_results: choose_slice(data,
                                                                                  load_results=load_results,
                                                                                  num_slice=2),
                                          load_results=load_results)

    field_1 = slice_1['line']['field']
    field_2 = slice_2['line']['field']
    peaks_field_range = (min(field_1, field_2),
                         max(field_1, field_2))
    data_filtered = data_io.filter_data_by_range(data=data,
                                                 x_range=peaks_field_range)
    
    if load_results:
        loaded_params = pickle.load(open(BUFFER_DATA_FILE, 'rb'))['found_peaks_data']
        output_params = {'found_peaks_data': loaded_params}
        return (data_filtered, output_params), output_params

    buttons = [["Select Peaks",
                ["selected_peaks",
                 lambda click, params, mode, ax: click_funcs.choose_points(click,
                                                                           params,
                                                                           mode,
                                                                           ax,
                                                                           num_points=2)]],
               ["Min Distance Between Peaks",
                ["min_distance",
                 lambda click,params, mode, ax: click_funcs.choose_points(click,
                                                                          params,
                                                                          mode,
                                                                          ax,
                                                                          num_points=2)]]]
    selector = interactive.plot_interactive_contour_map(data=data_filtered,
                                                        title='Select Peaks',
                                                        buttons=buttons)
    peaks = selector.get_params()['selected_peaks']
    min_distance = selector.get_params()['min_distance']
    min_distance = np.abs(min_distance[0][1] - min_distance[1][1]) #  in GHz
    peaks = [{"field": peak[0], "freq": peak[1]} for peak in peaks]

    peak_search_params = []

    for i, peak in enumerate(peaks):
        freqs = data_filtered['y']
        field_idx = np.argmin(np.abs(data_filtered['x'] - peak['field']))
        s_values = np.abs(data_filtered['z'][field_idx, :])

        line = {'x': freqs, 'y': s_values, 'label': f'S-parameter at Field {peak["field"]}'}
        line['x_label'] = 'Frequency (GHz)'
        line['y_label'] = 'S-parameter Magnitude'
        data_line = {'line': line, 'plot_type': 'one_line'}
        _, chosen_peak_params = choose_peaks(data_line,
                                             load_results=False,
                                             peak_num=i+1)

        current_prominence = chosen_peak_params[f'peak_{i+1}_params']['prominence'] # in S-parameter units
        current_width = chosen_peak_params[f'peak_{i+1}_params']['width'] # in GHz
        current_freq = chosen_peak_params[f'peak_{i+1}_params']['peak_freq'] # in GHz
        peak_search_params.append({'peak_freq': current_freq,
                                   'prominence': current_prominence,
                                   'width': current_width,
                                   'min_distance': min_distance})
    
    fields = data_filtered['x']

    found_modes = [{'freqs': [],
                    'fields': [],
                    'magnitudes': [],
                    'prominences': [],
                    'widths': []} for _ in range(len(peak_search_params))]

    for i, peak in enumerate(peak_search_params):
        for j, field in enumerate(fields):
            try:
                found_peak = useful_funcs.find_peak_flexible(
                    freqs=data_filtered['y'],
                    s_values=np.abs(data_filtered['z'][j, :]),
                    expected_freq=peak['peak_freq'],
                    expected_width=peak['width'],
                    expected_prominence=peak['prominence'],
                    search_window=peak['min_distance']
                )
                
                found_modes[i]['freqs'].append(found_peak['freq'])
                found_modes[i]['fields'].append(field)
                found_modes[i]['magnitudes'].append(found_peak['magnitude'])
                found_modes[i]['prominences'].append(found_peak['prominence'])
                found_modes[i]['widths'].append(found_peak['width'])
                
                peak_search_params[i]['peak_freq'] = found_peak['freq']
                peak_search_params[i]['width'] = found_peak['width']
                peak_search_params[i]['prominence'] = found_peak['prominence'] * 0.8  # С запасом
                
            except ValueError as e:
                print(f"[Peak {i+1}, Field {field:.1f}] Error: {e}")
                continue

    output_params = {'modes': found_modes,
                     'plot_type': 'peaks'}
    output_params = {'found_peaks_data': output_params}

    return (data_filtered, output_params), output_params

def approximate_peaks(data_tuple, load_results=False):
    if load_results:
        loaded_params = pickle.load(open(BUFFER_DATA_FILE, 'rb'))['fitted_peaks_data']
        output_params = {'fitted_peaks_data': loaded_params}
        full_data = data_tuple[0]
        return (full_data, output_params), output_params

    modes = data_tuple[1]['found_peaks_data']['modes']
    full_data = data_tuple[0]

    initial_params = []

    plt.ion()
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)
    plt.show(block=False)

    initial_params = [0, 0, 1.0, 0, 0]  # freq, width, q, a, offset

    for j, mode in enumerate(modes):
        fields = mode['fields']
        for i, field in enumerate(fields):
            freqs = full_data['y']
            field_idx = np.argmin(np.abs(full_data['x'] - field))
            s_values = np.abs(full_data['z'][field_idx, :])
            field_value = full_data['x'][field_idx]

            peak_freq = mode['freqs'][i]
            peak_width = mode['widths'][i]

            fit_region = (peak_freq - peak_width * 1.5, peak_freq + peak_width * 1.5)
            freq_indices = np.where((freqs >= fit_region[0]) & (freqs <= fit_region[1]))[0]

            local_freqs = freqs[freq_indices]
            local_s_values = s_values[freq_indices]
            amplitude_offset = np.min(local_s_values) if config_physics.PEAK_TYPE == 'maximum' else np.max(local_s_values)
            a = np.max(local_s_values) - amplitude_offset if config_physics.PEAK_TYPE == 'maximum' else amplitude_offset - np.min(local_s_values)

            initial_params[0] = peak_freq
            initial_params[1] = peak_width
            initial_params[3] = a
            initial_params[4] = amplitude_offset

            try:
                popt, _ = sp.optimize.curve_fit(useful_funcs.fano_model,
                                                local_freqs,
                                                local_s_values,
                                                p0=[peak_freq, peak_width, 1.0, a, amplitude_offset],
                                                maxfev=500000)
                
                fitted_freq = popt[0]
                fitted_width = popt[1]
                fitted_q = popt[2]
                fitted_a = popt[3]
                fitted_offset = popt[4]

                initial_params[2] = fitted_q

                modes[j]['freqs'][i] = fitted_freq
                modes[j]['widths'][i] = fitted_width
                modes[j]['amplitude_offset'] = fitted_offset
                modes[j]['fano_q'] = fitted_q
                modes[j]['fano_a'] = fitted_a
                modes[j]['field'] = field_value

            except Exception as e:
                print(f"[Peak {j+1}, Field {field:.1f}] Fitting Error: {e}")
                continue
            
            ax.clear()
            ax.set_title(f'Fitting Peak {j+1} at Field {field_value:.1f} Oe')
            ax.set_xlabel('Frequency (GHz)')
            ax.set_ylabel('S-parameter Magnitude')
            ax.plot(local_freqs, local_s_values, 'b.', label='Data')
            fitted_freqs = np.linspace(local_freqs[0], local_freqs[-1], 500)
            fitted_curve = useful_funcs.fano_model(fitted_freqs, *popt)
            ax.plot(fitted_freqs, fitted_curve, 'r-', label='Fitted Curve')
            ax.legend()
            plt.draw()
            plt.pause(0.1)  # Пауза
    
    plt.ioff()
    plt.close()
    output_params = {'modes': modes, 'plot_type': 'peaks'}
    output_params = {'fitted_peaks_data': output_params}
    return (full_data, output_params), output_params

def construct_own_modes(peaks):
    peak_modes = peaks['fitted_peaks_data']['modes']
    widths = []
    frequencies = []
    fields = []
    for i, mode in enumerate(peak_modes):
        width = np.array(mode['widths'])
        width = sp.signal.medfilt(width, 11)
        frequency = np.array(mode['freqs'])
        frequency = sp.signal.medfilt(frequency, 11)

        widths.append(width)
        frequencies.append(frequency)
        fields.append(mode['fields'])

    widths = np.array(widths)
    frequencies = np.array(frequencies)
    fields = np.array(fields)

    modes = []
    for i, width in enumerate(widths):
        mode_complex = frequencies[i] + 1j * width / 2
        modes.append(mode_complex)

    fig = plt.figure()
    ax = fig.subplots(2, 1)
    ax[0].set_title('Own Modes Real Part')
    for i, mode in enumerate(modes):
        ax[0].plot(fields[i], mode.real, label=f'Mode {i+1} Real')
    ax[0].legend()
    
    ax[1].set_title('Own Modes Imaginary Part')
    for i, mode in enumerate(modes):
        ax[1].plot(fields[i], mode.imag, label=f'Mode {i+1} Imag')
    ax[1].legend()

    ax[0].set_xlabel('Field')
    ax[0].set_ylabel('Frequency')

    ax[1].set_xlabel('Field')
    ax[1].set_ylabel('Frequency')
    plt.show()
    return {'modes': modes, 'fields': fields}

def extract_coupling_params(resonator, own_modes):
    resonator_params = resonator['resonator_params']
    output = []

    fields = own_modes['fields'][0]
    kappa = resonator_params['kappa']
    beta = resonator_params['beta']
    cavity_freq = resonator_params['resonance_freq']
    plato = resonator_params['plato']

    cavity_complex = cavity_freq - 1j * (beta + kappa)
    J, Gamma, gamma, magnon_freq, alpha = [], [], [], [], []

    mode_1 = own_modes['modes'][0]
    mode_2 = own_modes['modes'][1]

    # Сначала извлекаем экспериментальные частоты магнонов
    magnon_freq_experimental = []
    for i, field in enumerate(fields):
        mode_1_val = mode_1[i]
        mode_2_val = mode_2[i]
        magnon_complex = mode_1_val + mode_2_val - cavity_complex
        magnon_freq_experimental.append(magnon_complex.real)
    
    # Загружаем сдвиг из буферного файла
    try:
        buffer_data = pickle.load(open(BUFFER_DATA_FILE, 'rb'))
        offset = buffer_data.get('magnon_calibration', {}).get('offset', None)
        
        if offset is None:
            print("Предупреждение: сдвиг магнонной частоты не найден в буферном файле. Выполняется калибровка...")
            magnon_freq_calibrated, offset = useful_funcs.calibrate_magnon_frequency(
                fields, 
                np.array(magnon_freq_experimental)
            )
        else:
            # Используем сохраненный сдвиг для калибровки
            magnon_freq_calibrated = config_physics.GYROMAGNETIC_RATIO * fields + offset
            print(f"Используется сохраненный сдвиг магнонной частоты: {offset:.6f} ГГц")
    except Exception as e:
        print(f"Ошибка при загрузке сдвига из буферного файла: {e}")
        print("Выполняется калибровка...")
        magnon_freq_calibrated, offset = useful_funcs.calibrate_magnon_frequency(
            fields, 
            np.array(magnon_freq_experimental)
        )
    
    # Теперь рассчитываем параметры связи с использованием откалиброванных частот
    for i, field in enumerate(fields):
        mode_1_val = mode_1[i]
        mode_2_val = mode_2[i]
        
        # Используем откалиброванную частоту магнонов
        magnon_freq_calib = magnon_freq_calibrated[i]
        
        # Пересчитываем alpha с учетом откалиброванной частоты
        # magnon_complex = magnon_freq_calib - 1j * alpha
        # mode_1 + mode_2 = cavity_complex + magnon_complex
        # => magnon_complex = mode_1 + mode_2 - cavity_complex
        magnon_complex_from_modes = mode_1_val + mode_2_val - cavity_complex
        magnon_imag = magnon_complex_from_modes.imag
        
        # Создаем комплексную частоту магнона с откалиброванной частотой
        magnon_complex_calibrated = magnon_freq_calib + 1j * magnon_imag
        
        # Рассчитываем coupling из собственных мод, используя КАЛИБРОВАННЫЕ частоты
        # delta = 2 * cavity_complex - (magnon_complex_calibrated + cavity_complex)
        delta = cavity_complex - magnon_complex_calibrated
        coupling = np.sqrt((mode_1_val - mode_2_val)**2 - delta**2) / 2

        J_val = coupling.real
        Gamma_val = -coupling.imag
        gamma_val = Gamma_val**2 / kappa
        alpha_val = -magnon_complex_calibrated.imag - gamma_val

        J.append(np.abs(J_val))
        Gamma.append(np.abs(Gamma_val))
        gamma.append(np.abs(gamma_val))
        magnon_freq.append(np.abs(magnon_freq_calib))
        alpha.append(np.abs(alpha_val))

    params = {
        'field': fields,
        'J': J,
        'Gamma': Gamma,
        'gamma': gamma,
        'magnon_freq': magnon_freq,
        'magnon_freq_experimental': magnon_freq_experimental,
        'magnon_offset': offset,
        'cavity_freq': cavity_freq * np.ones_like(fields),
        'alpha': alpha,
        'kappa': kappa * np.ones_like(fields),
        'beta': beta * np.ones_like(fields),
        'plato': plato
    }

    return params

def plot_results(data, own_modes, coupling_params):

    fields = coupling_params['field']
    J_values = coupling_params['J']
    Gamma_values = coupling_params['Gamma']
    gamma_values = coupling_params['gamma']
    magnon_freq_values = coupling_params['magnon_freq']
    cavity_freq_values = coupling_params['cavity_freq']
    alpha_values = coupling_params['alpha']
    kappa_values = coupling_params['kappa']
    beta_values = coupling_params['beta']
    plato_values = coupling_params['plato']

    mode_1 = own_modes['modes'][0]
    mode_2 = own_modes['modes'][1]

    #Графики извлеченных параметров резонатора и связи
    fig = plt.figure(figsize=(12, 10))
    ax = fig.subplots(3, 2)
    
    # Уменьшаем размер шрифтов
    plt.rcParams.update({'font.size': 9})
    
    # График 1: Параметры резонатора
    ax[0, 0].plot(fields, kappa_values, 'b-', label='κ (external)')
    ax[0, 0].plot(fields, beta_values, 'r-', label='β (internal)')
    ax[0, 0].set_title('Resonator Loss Rates', fontsize=10, pad=8)
    ax[0, 0].set_xlabel('Magnetic Field (Oe)', fontsize=9)
    ax[0, 0].set_ylabel('Loss Rate (GHz)', fontsize=9)
    ax[0, 0].legend(fontsize=8)
    ax[0, 0].grid(True, alpha=0.3)

    # График 2: Параметры связи
    ax[0, 1].set_title('Coupling Parameters', fontsize=10, pad=8)
    ax[0, 1].plot(fields, J_values, 'g-', label='J (coherent)')
    ax[0, 1].plot(fields, Gamma_values, 'm-', label='Γ (dissipative)')
    ax[0, 1].set_xlabel('Magnetic Field (Oe)', fontsize=9)
    ax[0, 1].set_ylabel('Coupling Strength (GHz)', fontsize=9)
    ax[0, 1].legend(fontsize=8)
    ax[0, 1].grid(True, alpha=0.3)

    # График 3: Параметры магнонов
    ax[1, 0].set_title('Magnon Loss Rates', fontsize=10, pad=8)
    ax[1, 0].plot(fields, alpha_values, 'b-', label='α (intrinsic)')
    ax[1, 0].plot(fields, gamma_values, 'r-', label='γ (induced)')
    ax[1, 0].set_xlabel('Magnetic Field (Oe)', fontsize=9)
    ax[1, 0].set_ylabel('Loss Rate (GHz)', fontsize=9)
    ax[1, 0].legend(fontsize=8)
    ax[1, 0].grid(True, alpha=0.3)

    # График 4: Частичные частоты
    ax[1, 1].set_title('Bare Frequencies', fontsize=10, pad=8)
    ax[1, 1].plot(fields, magnon_freq_values, 'b-', label='Magnon')
    ax[1, 1].plot(fields, cavity_freq_values[0] * np.ones_like(fields), 'r--', label='Cavity')
    ax[1, 1].set_xlabel('Magnetic Field (Oe)', fontsize=9)
    ax[1, 1].set_ylabel('Frequency (GHz)', fontsize=9)
    ax[1, 1].legend(fontsize=8)
    ax[1, 1].grid(True, alpha=0.3)

    # График 5: Действительные части собственных мод
    ax[2, 0].set_title('Eigenmode Frequencies', fontsize=10, pad=8)
    ax[2, 0].plot(fields, mode_1.real, 'b-', label='Mode 1')
    ax[2, 0].plot(fields, mode_2.real, 'r-', label='Mode 2')
    ax[2, 0].set_xlabel('Magnetic Field (Oe)', fontsize=9)
    ax[2, 0].set_ylabel('Frequency (GHz)', fontsize=9)
    ax[2, 0].legend(fontsize=8)
    ax[2, 0].grid(True, alpha=0.3)

    # График 6: Мнимые части собственных мод
    ax[2, 1].set_title('Eigenmode Damping Rates', fontsize=10, pad=8)
    ax[2, 1].plot(fields, mode_1.imag, 'b-', label='Mode 1')
    ax[2, 1].plot(fields, mode_2.imag, 'r-', label='Mode 2')
    ax[2, 1].set_xlabel('Magnetic Field (Oe)', fontsize=9)
    ax[2, 1].set_ylabel('Damping Rate (GHz)', fontsize=9)
    ax[2, 1].legend(fontsize=8)
    ax[2, 1].grid(True, alpha=0.3)

    plt.tight_layout(pad=2.0)
    plt.savefig('results/resonator_parameters.png', dpi=150, bbox_inches='tight')
    plt.show()

def get_general_parameters(coupling_params):
    """
    Рассчитывает общие параметры системы как среднее значение с погрешностью
    
    Parameters:
    -----------
    coupling_params : dict
        Словарь с параметрами резонатора, связи и магнонов для разных полей
        Содержит ключи: 'kappa', 'beta', 'cavity_freq', 'plato', 
                       'J', 'Gamma', 'gamma', 'alpha', 'magnon_freq'
        
    Returns:
    --------
    general_params : dict
        Словарь с общими параметрами системы и их погрешностями:
        - Параметры резонатора (константы, без погрешности)
        - Средние значения параметров связи и магнонов
        - Стандартные отклонения как погрешности
    """
    # Параметры резонатора (константы для всех полей, берем первое значение)
    kappa_array = np.array(coupling_params['kappa'])
    beta_array = np.array(coupling_params['beta'])
    cavity_freq_array = np.array(coupling_params['cavity_freq'])
    
    general_params = {
        'kappa': kappa_array[0],
        'kappa_err': 0.0,
        'beta': beta_array[0],
        'beta_err': 0.0,
        'cavity_freq': cavity_freq_array[0],
        'cavity_freq_err': 0.0,
        'plato': coupling_params['plato'],
        'plato_err': 0.0
    }
    
    # Параметры связи - среднее и стандартное отклонение
    J_array = np.array(coupling_params['J'])
    general_params['J_mean'] = np.mean(J_array)
    general_params['J_err'] = np.std(J_array)
    
    Gamma_array = np.array(coupling_params['Gamma'])
    general_params['Gamma_mean'] = np.mean(Gamma_array)
    general_params['Gamma_err'] = np.std(Gamma_array)
    
    # Параметры магнонов - среднее и стандартное отклонение
    gamma_array = np.array(coupling_params['gamma'])
    general_params['gamma_mean'] = np.mean(gamma_array)
    general_params['gamma_err'] = np.std(gamma_array)
    
    alpha_array = np.array(coupling_params['alpha'])
    general_params['alpha_mean'] = np.mean(alpha_array)
    general_params['alpha_err'] = np.std(alpha_array)
    
    # Частота магнонов - среднее и стандартное отклонение
    magnon_freq_array = np.array(coupling_params['magnon_freq'])
    general_params['magnon_freq_mean'] = np.mean(magnon_freq_array)
    general_params['magnon_freq_err'] = np.std(magnon_freq_array)
    
    return general_params

def reconstruct_own_modes(general_params, coupling_params):
    fields = coupling_params['field']

    alpha = general_params['alpha_mean']
    beta = general_params['beta']
    gamma = general_params['gamma_mean']
    kappa = general_params['kappa']

    G = general_params['Gamma_mean']
    J = general_params['J_mean']

    magnon_freqs = coupling_params['magnon_freq']
    cavity_freq = general_params['cavity_freq']

    plato = general_params['plato']

    cavity_complex = cavity_freq - 1j * beta

    reconstructed_modes = {"mode_plus": [], "mode_minus": [], "fields": fields}
    for i, field in enumerate(fields):
        magnon_complex = magnon_freqs[i] - 1j * alpha
        coupling = J - 1j * G
        mode_plus = (cavity_complex + magnon_complex) / 2 + np.sqrt(coupling**2 + ((cavity_complex - magnon_complex) / 2)**2)
        mode_minus = (cavity_complex + magnon_complex) / 2 - np.sqrt(coupling**2 + ((cavity_complex - magnon_complex) / 2)**2)

        reconstructed_modes["mode_plus"].append(mode_plus)
        reconstructed_modes["mode_minus"].append(mode_minus)   
    return reconstructed_modes

def reconstruct_data(data, general_params, coupling_params):
    """
    Восстанавливает S-параметры по извлеченным параметрам и сравнивает с экспериментом
    
    Parameters:
    -----------
    data : dict
        Экспериментальные данные с ключами 'x' (поля), 'y' (частоты), 'z' (S-параметры)
    general_params : dict
        Общие параметры системы из get_general_parameters
    coupling_params : dict
        Параметры связи для каждого поля
        
    Returns:
    --------
    reconstructed_data : dict
        Словарь с восстановленными S-параметрами
    """
    # Извлекаем параметры
    alpha = general_params['alpha_mean']
    beta = general_params['beta']
    kappa = general_params['kappa']
    gamma = general_params['gamma_mean']
    
    J = general_params['J_mean']
    G = general_params['Gamma_mean']
    
    cavity_freq = general_params['cavity_freq']
    plato = general_params['plato']
    
    # Частоты и поля из экспериментальных данных
    freq = data['y']
    field = data['x']
    s_param_exp = data['z']
    
    # Частоты магнонов для каждого поля
    magnon_freqs = np.array(coupling_params['magnon_freq'])
    
    # Создаем матрицу S-параметров
    s_param_reconstructed = np.zeros((len(field), len(freq)), dtype=complex)
    
    # Рассчитываем S-параметр для каждого поля
    for i, (f, magnon_freq) in enumerate(zip(field, magnon_freqs)):
        cavity_delta = freq - cavity_freq
        magnon_delta = freq - magnon_freq
        coupling = J - 1j * G
        
        cavity_term = 1j * cavity_delta - (kappa + beta)
        magnon_term = 1j * magnon_delta - (gamma + alpha)
        
        response = plato + (kappa / (cavity_term - (coupling**2) / magnon_term))
        s_param_reconstructed[i, :] = response
    
    # Создаем график сравнения
    fig = plt.figure(figsize=(14, 6))
    
    # График 1: Экспериментальные данные
    ax1 = plt.subplot(1, 2, 1)
    s_exp_db = 20 * np.log10(np.abs(s_param_exp))
    contour1 = ax1.contourf(freq, field, s_exp_db, levels=50, cmap='viridis')
    ax1.set_xlabel('Frequency (GHz)', fontsize=11)
    ax1.set_ylabel('Magnetic Field (Oe)', fontsize=11)
    ax1.set_title('Experimental Data', fontsize=12, pad=10)
    cbar1 = plt.colorbar(contour1, ax=ax1)
    cbar1.set_label('|S21| (dB)', fontsize=10)
    
    # График 2: Восстановленные данные
    ax2 = plt.subplot(1, 2, 2)
    s_rec_db = 20 * np.log10(np.abs(s_param_reconstructed))
    contour2 = ax2.contourf(freq, field, s_rec_db, levels=50, cmap='viridis')
    ax2.set_xlabel('Frequency (GHz)', fontsize=11)
    ax2.set_ylabel('Magnetic Field (Oe)', fontsize=11)
    ax2.set_title('Reconstructed Data', fontsize=12, pad=10)
    cbar2 = plt.colorbar(contour2, ax=ax2)
    cbar2.set_label('|S21| (dB)', fontsize=10)
    
    plt.tight_layout(pad=2.0)
    plt.savefig('results/data_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    reconstructed_data = {
        'freq': freq,
        'field': field,
        's_param': s_param_reconstructed
    }
    
    return reconstructed_data

def visualize_approximated_modes(reconstructed_modes, experimental_modes):
    """
    Сравнивает экспериментальные и восстановленные собственные моды
    
    Parameters:
    -----------
    reconstructed_modes : dict
        Словарь с восстановленными модами из reconstruct_own_modes
        Содержит: 'mode_plus', 'mode_minus', 'fields'
    experimental_modes : dict
        Словарь с экспериментальными модами из construct_own_modes
        Содержит: 'modes', 'fields'
    """
    fields = reconstructed_modes['fields']
    
    # Экспериментальные моды
    exp_mode_1 = experimental_modes['modes'][0]
    exp_mode_2 = experimental_modes['modes'][1]
    
    # Восстановленные моды
    rec_mode_plus = np.array(reconstructed_modes['mode_plus'])
    rec_mode_minus = np.array(reconstructed_modes['mode_minus'])
    
    # Создаем график
    fig = plt.figure(figsize=(14, 10))
    
    # Уменьшаем размер шрифтов
    plt.rcParams.update({'font.size': 9})
    
    # График 1: Действительные части (частоты)
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(fields, exp_mode_1.real, 'b-', linewidth=2, label='Exp. Mode 1')
    ax1.plot(fields, exp_mode_2.real, 'r-', linewidth=2, label='Exp. Mode 2')
    ax1.plot(fields, rec_mode_plus.real, 'b--', linewidth=1.5, label='Rec. Mode +')
    ax1.plot(fields, rec_mode_minus.real, 'r--', linewidth=1.5, label='Rec. Mode −')
    ax1.set_xlabel('Magnetic Field (Oe)', fontsize=10)
    ax1.set_ylabel('Frequency (GHz)', fontsize=10)
    ax1.set_title('Eigenmode Frequencies Comparison', fontsize=11, pad=10)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # График 2: Мнимые части (затухание)
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(fields, exp_mode_1.imag, 'b-', linewidth=2, label='Exp. Mode 1')
    ax2.plot(fields, exp_mode_2.imag, 'r-', linewidth=2, label='Exp. Mode 2')
    ax2.plot(fields, rec_mode_plus.imag, 'b--', linewidth=1.5, label='Rec. Mode +')
    ax2.plot(fields, rec_mode_minus.imag, 'r--', linewidth=1.5, label='Rec. Mode −')
    ax2.set_xlabel('Magnetic Field (Oe)', fontsize=10)
    ax2.set_ylabel('Damping Rate (GHz)', fontsize=10)
    ax2.set_title('Eigenmode Damping Rates Comparison', fontsize=11, pad=10)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # График 3: Разность частот (Mode 1 vs Mode +)
    ax3 = plt.subplot(2, 2, 3)
    diff_freq_1 = exp_mode_1.real - rec_mode_plus.real
    diff_freq_2 = exp_mode_2.real - rec_mode_minus.real
    ax3.plot(fields, diff_freq_1 * 1000, 'b-', linewidth=2, label='Mode 1 − Mode +')
    ax3.plot(fields, diff_freq_2 * 1000, 'r-', linewidth=2, label='Mode 2 − Mode −')
    ax3.axhline(y=0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    ax3.set_xlabel('Magnetic Field (Oe)', fontsize=10)
    ax3.set_ylabel('Frequency Difference (MHz)', fontsize=10)
    ax3.set_title('Frequency Residuals', fontsize=11, pad=10)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # График 4: Разность затуханий
    ax4 = plt.subplot(2, 2, 4)
    diff_damp_1 = exp_mode_1.imag - rec_mode_plus.imag
    diff_damp_2 = exp_mode_2.imag - rec_mode_minus.imag
    ax4.plot(fields, diff_damp_1 * 1000, 'b-', linewidth=2, label='Mode 1 − Mode +')
    ax4.plot(fields, diff_damp_2 * 1000, 'r-', linewidth=2, label='Mode 2 − Mode −')
    ax4.axhline(y=0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    ax4.set_xlabel('Magnetic Field (Oe)', fontsize=10)
    ax4.set_ylabel('Damping Difference (MHz)', fontsize=10)
    ax4.set_title('Damping Residuals', fontsize=11, pad=10)
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout(pad=2.0)
    plt.savefig('results/modes_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Выводим статистику
    print("\n" + "="*60)
    print("COMPARISON STATISTICS")
    print("="*60)
    print(f"\nFrequency Residuals (RMS):")
    print(f"  Mode 1 vs Mode +: {np.sqrt(np.mean(diff_freq_1**2))*1000:.4f} MHz")
    print(f"  Mode 2 vs Mode −: {np.sqrt(np.mean(diff_freq_2**2))*1000:.4f} MHz")
    print(f"\nDamping Residuals (RMS):")
    print(f"  Mode 1 vs Mode +: {np.sqrt(np.mean(diff_damp_1**2))*1000:.4f} MHz")
    print(f"  Mode 2 vs Mode −: {np.sqrt(np.mean(diff_damp_2**2))*1000:.4f} MHz")
    print("="*60 + "\n")

def visualize_approximated_data(reconstructed_data, original_data):
    plt.figure()
    plt.title('Approximated Data vs Original Data')

    freq = reconstructed_data['freq']
    field = reconstructed_data['field']
    s_param_reconstructed = reconstructed_data['s_param']
    s_param_original = original_data['s_param']

    plt.subplot(2, 1, 1)
    plt.title('Reconstructed S-Parameter Magnitude')
    plt.contourf(np.abs(s_param_reconstructed), extent=[freq[0], freq[-1], field[0], field[-1]], aspect='auto', origin='lower')
    plt.colorbar(label='|S-Parameter|')
    plt.xlabel('Frequency')
    plt.ylabel('Field')

    plt.subplot(2, 1, 2)
    plt.title('Original S-Parameter Magnitude')
    plt.contourf(np.abs(s_param_original), extent=[freq[0], freq[-1], field[0], field[-1]], aspect='auto', origin='lower')
    plt.colorbar(label='|S-Parameter|')
    plt.xlabel('Frequency')
    plt.ylabel('Field')

    plt.savefig('results/approximated_data.png')
    plt.show()

def save_results(coupling_params, general_params, reconstructed_data, data, own_modes):
    """
    Сохраняет результаты анализа в отдельную подпапку с временной меткой
    
    Parameters:
    -----------
    coupling_params : dict
        Параметры связи для каждого поля
    general_params : dict
        Усредненные параметры системы
    reconstructed_data : dict
        Восстановленные S-параметры
    data : dict
        Экспериментальные данные
    own_modes : dict
        Собственные моды системы
    """
    from datetime import datetime
    import os
    
    # Создаем папку с временной меткой
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = os.path.join('results', timestamp)
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"\nСохранение результатов в: {results_dir}")
    
    # ========================================================================
    # 1. СОХРАНЕНИЕ ПАРАМЕТРОВ В ТЕКСТОВЫЕ ФАЙЛЫ
    # ========================================================================
    
    # Файл с усредненными параметрами
    with open(os.path.join(results_dir, 'general_parameters.txt'), 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("УСРЕДНЕННЫЕ ПАРАМЕТРЫ СИСТЕМЫ\n")
        f.write("="*70 + "\n\n")
        
        f.write("ПАРАМЕТРЫ РЕЗОНАТОРА:\n")
        f.write("-"*70 + "\n")
        f.write(f"Частота резонатора:     ω_c = {general_params['cavity_freq']:.6f} ГГц\n")
        f.write(f"Внешнее затухание:      κ   = {general_params['kappa']:.6f} ГГц\n")
        f.write(f"Внутреннее затухание:   β   = {general_params['beta']:.6f} ГГц\n")
        f.write(f"Плато S-параметра:      S₀  = {general_params['plato']:.6f}\n\n")
        
        f.write("ПАРАМЕТРЫ СВЯЗИ:\n")
        f.write("-"*70 + "\n")
        f.write(f"Когерентная связь:      J   = {general_params['J_mean']:.6f} ± {general_params['J_err']:.6f} ГГц\n")
        f.write(f"Диссипативная связь:    Γ   = {general_params['Gamma_mean']:.6f} ± {general_params['Gamma_err']:.6f} ГГц\n\n")
        
        f.write("ПАРАМЕТРЫ МАГНОНОВ:\n")
        f.write("-"*70 + "\n")
        f.write(f"Средняя частота:        ω_m = {general_params['magnon_freq_mean']:.6f} ± {general_params['magnon_freq_err']:.6f} ГГц\n")
        f.write(f"Собственное затухание:  α   = {general_params['alpha_mean']:.6f} ± {general_params['alpha_err']:.6f} ГГц\n")
        f.write(f"Индуцированное затух.:  γ   = {general_params['gamma_mean']:.6f} ± {general_params['gamma_err']:.6f} ГГц\n")
        f.write(f"Калибровочное смещение: Δω  = {coupling_params['magnon_offset']:.6f} ГГц\n\n")
        
        f.write("="*70 + "\n")
    
    # Файл с параметрами по полям
    fields = coupling_params['field']
    with open(os.path.join(results_dir, 'field_dependent_parameters.txt'), 'w', encoding='utf-8') as f:
        f.write("="*120 + "\n")
        f.write("ПАРАМЕТРЫ СВЯЗИ И МАГНОНОВ ДЛЯ КАЖДОГО ЗНАЧЕНИЯ МАГНИТНОГО ПОЛЯ\n")
        f.write("="*120 + "\n\n")
        
        header = f"{'Поле (Э)':>12} {'ω_m (ГГц)':>12} {'ω_m_эксп':>12} {'J (ГГц)':>12} {'Γ (ГГц)':>12} {'α (ГГц)':>12} {'γ (ГГц)':>12}\n"
        f.write(header)
        f.write("-"*120 + "\n")
        
        for i, field in enumerate(fields):
            line = f"{field:12.2f} {coupling_params['magnon_freq'][i]:12.6f} {coupling_params['magnon_freq_experimental'][i]:12.6f} "
            line += f"{coupling_params['J'][i]:12.6f} {coupling_params['Gamma'][i]:12.6f} "
            line += f"{coupling_params['alpha'][i]:12.6f} {coupling_params['gamma'][i]:12.6f}\n"
            f.write(line)
    
    # ========================================================================
    # 2. ГРАФИК: ПАРАМЕТРЫ РЕЗОНАТОРА И СВЯЗИ
    # ========================================================================
    
    fig = plt.figure(figsize=(14, 10))
    plt.rcParams.update({'font.size': 10})
    
    mode_1 = own_modes['modes'][0]
    mode_2 = own_modes['modes'][1]
    
    # График 1: Параметры резонатора
    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(fields, coupling_params['kappa'], 'b-', linewidth=2, label='κ (external)')
    ax1.plot(fields, coupling_params['beta'], 'r-', linewidth=2, label='β (internal)')
    ax1.axhline(y=general_params['kappa'], color='b', linestyle='--', alpha=0.5)
    ax1.axhline(y=general_params['beta'], color='r', linestyle='--', alpha=0.5)
    ax1.set_title('Resonator Loss Rates', fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlabel('Magnetic Field (Oe)', fontsize=10)
    ax1.set_ylabel('Loss Rate (GHz)', fontsize=10)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # График 2: Параметры связи
    ax2 = plt.subplot(3, 2, 2)
    ax2.plot(fields, coupling_params['J'], 'g-', linewidth=2, label=f'J = {general_params["J_mean"]:.4f}±{general_params["J_err"]:.4f} GHz')
    ax2.plot(fields, coupling_params['Gamma'], 'm-', linewidth=2, label=f'Γ = {general_params["Gamma_mean"]:.4f}±{general_params["Gamma_err"]:.4f} GHz')
    ax2.axhline(y=general_params['J_mean'], color='g', linestyle='--', alpha=0.5)
    ax2.axhline(y=general_params['Gamma_mean'], color='m', linestyle='--', alpha=0.5)
    ax2.set_title('Coupling Parameters', fontsize=12, fontweight='bold', pad=10)
    ax2.set_xlabel('Magnetic Field (Oe)', fontsize=10)
    ax2.set_ylabel('Coupling Strength (GHz)', fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # График 3: Параметры магнонов
    ax3 = plt.subplot(3, 2, 3)
    ax3.plot(fields, coupling_params['alpha'], 'b-', linewidth=2, label=f'α = {general_params["alpha_mean"]:.4f}±{general_params["alpha_err"]:.4f} GHz')
    ax3.plot(fields, coupling_params['gamma'], 'r-', linewidth=2, label=f'γ = {general_params["gamma_mean"]:.4f}±{general_params["gamma_err"]:.4f} GHz')
    ax3.axhline(y=general_params['alpha_mean'], color='b', linestyle='--', alpha=0.5)
    ax3.axhline(y=general_params['gamma_mean'], color='r', linestyle='--', alpha=0.5)
    ax3.set_title('Magnon Loss Rates', fontsize=12, fontweight='bold', pad=10)
    ax3.set_xlabel('Magnetic Field (Oe)', fontsize=10)
    ax3.set_ylabel('Loss Rate (GHz)', fontsize=10)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # График 4: Частоты (калиброванные vs экспериментальные)
    ax4 = plt.subplot(3, 2, 4)
    ax4.plot(fields, coupling_params['magnon_freq'], 'b-', linewidth=2, label='Magnon (calibrated)')
    ax4.plot(fields, coupling_params['magnon_freq_experimental'], 'b:', linewidth=1.5, alpha=0.7, label='Magnon (experimental)')
    ax4.axhline(y=general_params['cavity_freq'], color='r', linestyle='--', linewidth=2, label='Cavity')
    ax4.set_title('Bare Frequencies', fontsize=12, fontweight='bold', pad=10)
    ax4.set_xlabel('Magnetic Field (Oe)', fontsize=10)
    ax4.set_ylabel('Frequency (GHz)', fontsize=10)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # График 5: Действительные части собственных мод
    ax5 = plt.subplot(3, 2, 5)
    ax5.plot(fields, mode_1.real, 'b-', linewidth=2, label='Mode 1')
    ax5.plot(fields, mode_2.real, 'r-', linewidth=2, label='Mode 2')
    ax5.set_title('Eigenmode Frequencies', fontsize=12, fontweight='bold', pad=10)
    ax5.set_xlabel('Magnetic Field (Oe)', fontsize=10)
    ax5.set_ylabel('Frequency (GHz)', fontsize=10)
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    
    # График 6: Мнимые части собственных мод
    ax6 = plt.subplot(3, 2, 6)
    ax6.plot(fields, mode_1.imag, 'b-', linewidth=2, label='Mode 1')
    ax6.plot(fields, mode_2.imag, 'r-', linewidth=2, label='Mode 2')
    ax6.set_title('Eigenmode Damping Rates', fontsize=12, fontweight='bold', pad=10)
    ax6.set_xlabel('Magnetic Field (Oe)', fontsize=10)
    ax6.set_ylabel('Damping Rate (GHz)', fontsize=10)
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout(pad=2.0)
    plt.savefig(os.path.join(results_dir, 'system_parameters.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # ========================================================================
    # 3. ГРАФИК: СРАВНЕНИЕ ЭКСПЕРИМЕНТАЛЬНЫХ И ВОССТАНОВЛЕННЫХ ДАННЫХ
    # ========================================================================
    
    fig = plt.figure(figsize=(16, 6))
    
    freq = reconstructed_data['freq']
    field = reconstructed_data['field']
    s_param_exp = data['z']
    s_param_rec = reconstructed_data['s_param']
    
    # Экспериментальные данные
    ax1 = plt.subplot(1, 3, 1)
    s_exp_db = 20 * np.log10(np.abs(s_param_exp))
    contour1 = ax1.contourf(freq, field, s_exp_db, levels=100, cmap='viridis')
    ax1.set_xlabel('Frequency (GHz)', fontsize=11)
    ax1.set_ylabel('Magnetic Field (Oe)', fontsize=11)
    ax1.set_title('Experimental Data', fontsize=12, fontweight='bold', pad=10)
    cbar1 = plt.colorbar(contour1, ax=ax1)
    cbar1.set_label('|S₂₁| (dB)', fontsize=10)
    
    # Восстановленные данные
    ax2 = plt.subplot(1, 3, 2)
    s_rec_db = 20 * np.log10(np.abs(s_param_rec))
    contour2 = ax2.contourf(freq, field, s_rec_db, levels=100, cmap='viridis')
    ax2.set_xlabel('Frequency (GHz)', fontsize=11)
    ax2.set_ylabel('Magnetic Field (Oe)', fontsize=11)
    ax2.set_title('Reconstructed Data', fontsize=12, fontweight='bold', pad=10)
    cbar2 = plt.colorbar(contour2, ax=ax2)
    cbar2.set_label('|S₂₁| (dB)', fontsize=10)
    
    # Разность
    ax3 = plt.subplot(1, 3, 3)
    diff_db = s_exp_db - s_rec_db
    contour3 = ax3.contourf(freq, field, diff_db, levels=100, cmap='RdBu_r', vmin=-np.max(np.abs(diff_db)), vmax=np.max(np.abs(diff_db)))
    ax3.set_xlabel('Frequency (GHz)', fontsize=11)
    ax3.set_ylabel('Magnetic Field (Oe)', fontsize=11)
    ax3.set_title('Difference (Exp - Rec)', fontsize=12, fontweight='bold', pad=10)
    cbar3 = plt.colorbar(contour3, ax=ax3)
    cbar3.set_label('Δ|S₂₁| (dB)', fontsize=10)
    
    plt.tight_layout(pad=2.0)
    plt.savefig(os.path.join(results_dir, 'data_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # ========================================================================
    # 4. ГРАФИК: СРАВНЕНИЕ СОБСТВЕННЫХ МОД
    # ========================================================================
    
    # Восстанавливаем моды для сравнения
    reconstructed_modes = reconstruct_own_modes(general_params, coupling_params)
    
    fig = plt.figure(figsize=(14, 10))
    
    exp_mode_1 = own_modes['modes'][0]
    exp_mode_2 = own_modes['modes'][1]
    rec_mode_plus = np.array(reconstructed_modes['mode_plus'])
    rec_mode_minus = np.array(reconstructed_modes['mode_minus'])
    
    # График 1: Частоты
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(fields, exp_mode_1.real, 'b-', linewidth=2.5, label='Exp. Mode 1', alpha=0.8)
    ax1.plot(fields, exp_mode_2.real, 'r-', linewidth=2.5, label='Exp. Mode 2', alpha=0.8)
    ax1.plot(fields, rec_mode_plus.real, 'b--', linewidth=1.5, label='Rec. Mode +')
    ax1.plot(fields, rec_mode_minus.real, 'r--', linewidth=1.5, label='Rec. Mode −')
    ax1.set_xlabel('Magnetic Field (Oe)', fontsize=11)
    ax1.set_ylabel('Frequency (GHz)', fontsize=11)
    ax1.set_title('Eigenmode Frequencies', fontsize=12, fontweight='bold', pad=10)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # График 2: Затухания
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(fields, mode_1.imag, 'b-', linewidth=2.5, label='Exp. Mode 1', alpha=0.8)
    ax2.plot(fields, mode_2.imag, 'r-', linewidth=2.5, label='Exp. Mode 2', alpha=0.8)
    ax2.plot(fields, rec_mode_plus.imag, 'b--', linewidth=1.5, label='Rec. Mode +')
    ax2.plot(fields, rec_mode_minus.imag, 'r--', linewidth=1.5, label='Rec. Mode −')
    ax2.set_xlabel('Magnetic Field (Oe)', fontsize=11)
    ax2.set_ylabel('Damping Rate (GHz)', fontsize=11)
    ax2.set_title('Eigenmode Damping Rates', fontsize=12, fontweight='bold', pad=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # График 3: Разность частот
    ax3 = plt.subplot(2, 2, 3)
    diff_freq_1 = (exp_mode_1.real - rec_mode_plus.real) * 1000
    diff_freq_2 = (exp_mode_2.real - rec_mode_minus.real) * 1000
    ax3.plot(fields, diff_freq_1, 'b-', linewidth=2, label=f'Mode 1 − Mode + (RMS={np.sqrt(np.mean(diff_freq_1**2)):.2f} MHz)')
    ax3.plot(fields, diff_freq_2, 'r-', linewidth=2, label=f'Mode 2 − Mode − (RMS={np.sqrt(np.mean(diff_freq_2**2)):.2f} MHz)')
    ax3.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax3.set_xlabel('Magnetic Field (Oe)', fontsize=11)
    ax3.set_ylabel('Frequency Difference (MHz)', fontsize=11)
    ax3.set_title('Frequency Residuals', fontsize=12, fontweight='bold', pad=10)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # График 4: Разность затуханий
    ax4 = plt.subplot(2, 2, 4)
    diff_damp_1 = (exp_mode_1.imag - rec_mode_plus.imag) * 1000
    diff_damp_2 = (exp_mode_2.imag - rec_mode_minus.imag) * 1000
    ax4.plot(fields, diff_damp_1, 'b-', linewidth=2, label=f'Mode 1 − Mode + (RMS={np.sqrt(np.mean(diff_damp_1**2)):.2f} MHz)')
    ax4.plot(fields, diff_damp_2, 'r-', linewidth=2, label=f'Mode 2 − Mode − (RMS={np.sqrt(np.mean(diff_damp_2**2)):.2f} MHz)')
    ax4.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax4.set_xlabel('Magnetic Field (Oe)', fontsize=11)
    ax4.set_ylabel('Damping Difference (MHz)', fontsize=11)
    ax4.set_title('Damping Residuals', fontsize=12, fontweight='bold', pad=10)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout(pad=2.0)
    plt.savefig(os.path.join(results_dir, 'modes_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # ========================================================================
    # 5. СОХРАНЕНИЕ ДАННЫХ СОБСТВЕННЫХ МОД
    # ========================================================================
    
    with open(os.path.join(results_dir, 'eigenmodes_data.txt'), 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write("ДАННЫЕ СОБСТВЕННЫХ МОД\n")
        f.write("="*100 + "\n\n")
        
        header = f"{'Поле (Э)':>12} {'Mode1_Re':>12} {'Mode1_Im':>12} {'Mode2_Re':>12} {'Mode2_Im':>12} "
        header += f"{'Rec+_Re':>12} {'Rec+_Im':>12} {'Rec-_Re':>12} {'Rec-_Im':>12}\n"
        f.write(header)
        f.write("-"*100 + "\n")
        
        for i, field in enumerate(fields):
            line = f"{field:12.2f} {exp_mode_1.real[i]:12.6f} {exp_mode_1.imag[i]:12.6f} "
            line += f"{exp_mode_2.real[i]:12.6f} {exp_mode_2.imag[i]:12.6f} "
            line += f"{rec_mode_plus.real[i]:12.6f} {rec_mode_plus.imag[i]:12.6f} "
            line += f"{rec_mode_minus.real[i]:12.6f} {rec_mode_minus.imag[i]:12.6f}\n"
            f.write(line)
    
    print(f"✓ Сохранены параметры системы")
    print(f"✓ Сохранены графики (3 файла PNG, 300 DPI)")
    print(f"✓ Сохранены данные в текстовых файлах (3 файла TXT)")
    print(f"\nВсе результаты в папке: {results_dir}\n")

if __name__ == "__main__":
    data = data_io.load_s_parameter_data(config_data.FILEPATH)

    data, preprocess_params = check_prosessing_results(data, preprocess_data, load_results=False)

    _, magnon_calibration_params = magnon_calibration(data, load_results=False)

    _, resonator = check_prosessing_results(data, extract_resonator_params, load_results=False)
    (peaks_data, _), peaks = check_prosessing_results(data, find_peaks, load_results=False)
    (peaks_data, _), peaks_fitted = check_prosessing_results((peaks_data, peaks), approximate_peaks, load_results=False)

    own_modes = construct_own_modes(peaks_fitted)

    coupling_params = extract_coupling_params(resonator, own_modes)

    plot_results(data, own_modes, coupling_params)

    general_params = get_general_parameters(coupling_params)

    reconstructed_modes = reconstruct_own_modes(general_params, coupling_params)

    visualize_approximated_modes(reconstructed_modes, own_modes)

    reconstructed_data = reconstruct_data(data, general_params, coupling_params)
    
    save_results(coupling_params, general_params, reconstructed_data, data, own_modes)


# peaks = find_peaks(data)

# own_modes = construct_own_modes(peaks)

# coupling_params = extract_coupling_params(resonator, own_modes)

# plot_results(data, peaks, resonator, own_modes, coupling_params)

# save_results(peaks, resonator, own_modes, coupling_params, 'results/analysis_output.pkl')

# reconstructed_modes = reconstruct_own_modes(resonator, coupling_params, data['field'])

# reconstructed_data = reconstruct_data(resonator, coupling_params, data['freq'], data['field'])

# visualize_approximated_modes(reconstructed_modes, own_modes)

# visualize_approximated_data(reconstructed_data, data)
