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
import interactive2

BUFFER_DATA_FILE = "bufer_data.pkl"

def check_prosessing_results(data, proc_func, load_results= False):
    if load_results:
        output = proc_func(data, load_results=load_results)
        return output

    while True:
        output = proc_func(data, load_results=load_results)
        buttons = [["Confirm (double click)", ["true_func", click_funcs.true_func]],
                   ["Re-process (double click)", ["false_func", click_funcs.false_func]]]
        selector = interactive2.plot_interactive_contour_map(data=output,
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
        data_range = pickle.load(open('selected_params.pkl', 'rb'))["data_range"]
        data = data_io.filter_data_by_range(data=data,
                                            freq_range=data_range['freq_range'],
                                            field_range=data_range['field_range'])
        return data, {"data_range": data_range}
    
    buttons = [["Select Range", ["data_range", click_funcs.choose_range]]]
    selector = interactive2.plot_interactive_contour_map(data=data,
                                                        buttons=buttons,
                                                        title='Select Data Range',
                                                        plot_type=data['plot_type'])

    data_range = selector.get_params()["data_range"]

    data_filtered = data_io.filter_data_by_range(data=data,
                                        freq_range=data_range['freq_range'],
                                        field_range=data_range['field_range'])

    return data_filtered, {"data_range": data_range}

def extract_resonator_params(data, load_results=False):
    loaded_params = pickle.load(open('selected_params.pkl', 'rb'))['resonator'] if load_results else None
    if load_results:
        resonator_fit_region = loaded_params['cavity_fit_region']
        resonance_freq = loaded_params['cavity_frequency']
        cavity_width = loaded_params['cavity_width']
    else:
        buttons = [["Select Cavity Fit Region", ["cavity_fit_region", click_funcs.choose_range]],
                ["Select Cavity Frequency", ["cavity_frequency", click_funcs.choose_point]],
                ["Select Cavity Width", ["cavity_width", lambda click, params, mode, ax: click_funcs.choose_points(click, params, mode, ax, num_points=2)]]]
        selector = interactive2.plot_interactive_contour_map(data=data,
                                                            buttons=buttons,
                                                            title='Select Resonator Parameters',
                                                            plot_type=data['plot_type'])
        
        selected_resonator_params = selector.get_params()

        resonator_fit_region = selected_resonator_params['cavity_fit_region']
        resonance_freq = selected_resonator_params['cavity_frequency'][1]
        cavity_width = np.abs(selected_resonator_params['cavity_width'][0][1] - selected_resonator_params['cavity_width'][1][1])

    filtered_data = data_io.filter_data_by_range(data=data,
                                                 freq_range=resonator_fit_region['freq_range'],
                                                 field_range=resonator_fit_region['field_range'])

    freqs = filtered_data['freq']
    s_value = filtered_data['s_param']

    s_average = np.mean(np.abs(s_value), axis=0)
    res_index = np.argmin(np.abs(freqs - resonance_freq))

    res_magnitude = s_average[res_index]
    resonance_freq = freqs[res_index]
    plato = np.max(s_average) if config_physics.PEAK_TYPE == 'minimum' else np.min(s_average)

    estimated_params = useful_funcs.estimate_cavity_params(res_magnitude,
                                                           resonance_freq,
                                                           cavity_width, plato)
    
    fitted_params = useful_funcs.fit_cavity_response(freqs, s_average, estimated_params)

    output_data = {}
    experimental_line = {'x': freqs, 'y': s_average, 'label': 'experimental average'}
    modeled_s_params = np.abs(useful_funcs.cavity_model(freqs,
                                                        fitted_params['kappa'],
                                                        fitted_params['beta'],
                                                        fitted_params['resonance_freq'],
                                                        fitted_params['plato']))
    modeled_line = {'x': freqs, 'y': modeled_s_params, 'label': 'fitted model'}
    
    experimental_line['x_label'] = 'Frequency (GHz)'
    experimental_line['y_label'] = 'S-parameter Magnitude'
    modeled_line['x_label'] = 'Frequency (GHz)'
    modeled_line['y_label'] = 'S-parameter Magnitude'

    output_data['line1'] = experimental_line
    output_data['line2'] = modeled_line
    output_data['plot_type'] = 'two_line'

    output_params = {
        'kappa': fitted_params['kappa'],
        'beta': fitted_params['beta'],
        'resonance_freq': fitted_params['resonance_freq'],
        'plato': fitted_params['plato'],
        'res_magnitude': fitted_params['res_magnitude'],
    }
    output_params.update(output_data)

    return output_data, {'resonator': output_params}

def choose_slice(data, load_results=False, num_slice=1):
    if load_results:
        slice_params = pickle.load(open('selected_params.pkl', 'rb'))[f'slice_{num_slice}']
        return slice_params['slice_data'], slice_params

    buttons = [["Select Slice", ["slice_point", click_funcs.choose_point]]]
    selector = interactive2.plot_interactive_contour_map(data=data,
                                                        title='Select Slice',
                                                        buttons=buttons,
                                                        plot_type=data['plot_type'])
    
    slice_point = selector.get_params()['slice_point']
    slice_point = {'field': slice_point[0], 'freq': slice_point[1]}

    field_idx = np.argmin(np.abs(data['field'] - slice_point['field']))
    s_values = np.abs(data['s_param'][field_idx, :])
    freqs = data['freq']

    line = {'x': freqs, 'y': s_values, 'label': f'S-parameter at Field {slice_point["field"]}', 'x_label': 'Frequency (GHz)', 'y_label': 'S-parameter Magnitude', 'plot_type': 'one_line', 'field': slice_point['field']}

    slice_data = {'line': line, 'plot_type': line['plot_type']}
    params = {'slice_point': slice_point, 'slice_data': slice_data}
    return slice_data, {f'slice_{num_slice}': params}

def choose_peaks(data, load_results=False):
    if load_results:
        pass
    buttons = [["Select Prominence", ["prominence", lambda click, params, mode, ax: click_funcs.choose_points(click, params, mode, ax, num_points=2)]],
                ["Select Width", ["width", lambda click, params, mode, ax: click_funcs.choose_points(click, params, mode, ax, num_points=2)]],
                ["Peak frequency", ["peak_freq", click_funcs.choose_point]]]
    selector = interactive2.plot_interactive_contour_map(data=data,
                                                        title=f'Select Prominence for Peak',
                                                        buttons=buttons,
                                                        plot_type=data['plot_type'])
    prominence_points = selector.get_params()['prominence']
    width_points = selector.get_params()['width']
    peak_freq = selector.get_params()['peak_freq'][0]
    current_peak = {'prominence': np.abs(prominence_points[0][1] - prominence_points[1][1]), 'width': np.abs(width_points[0][0] - width_points[1][0]), 'peak_freq': peak_freq}
    return data, {'current_peak': current_peak}

def find_peaks(data, load_results=False):
    if load_results:
        peaks_data = pickle.load(open('selected_params.pkl', 'rb'))['peaks_data']
        return peaks_data, peaks_data

    slice_1 = check_prosessing_results(data, lambda data, load_results: choose_slice(data, load_results=load_results, num_slice=1), load_results=True)
    slice_2 = check_prosessing_results(data, lambda data, load_results: choose_slice(data, load_results=load_results, num_slice=2), load_results=True)

    field_1 = slice_1['line']['field']
    field_2 = slice_2['line']['field']
    data = data_io.filter_data_by_range(data=data,
                                        field_range=(field_1, field_2))

    buttons = [["Select Peaks", ["selected_peaks", lambda click, params, mode, ax: click_funcs.choose_points(click, params, mode, ax, num_points=2)]], ["Min Distance Between Peaks", ["min_distance", lambda click, params, mode, ax: click_funcs.choose_points(click, params, mode, ax, num_points=2)]]]
    selector = interactive2.plot_interactive_contour_map(data=data,
                                                        title='Select Peaks',
                                                        buttons=buttons,
                                                        plot_type=data['plot_type'])
    peaks = selector.get_params()['selected_peaks']
    min_distance = selector.get_params()['min_distance']
    min_distance = np.abs(min_distance[0][1] - min_distance[1][1]) #  in GHz
    peaks = [{"field": peak[0], "freq": peak[1]} for peak in peaks]

    peak_search_params = []

    for peak in peaks:
        freqs = data['freq']
        field_idx = np.argmin(np.abs(data['field'] - peak['field']))
        s_values = np.abs(data['s_param'][field_idx, :])

        line = {'x': freqs, 'y': s_values, 'label': f'S-parameter at Field {peak["field"]}'}
        line['x_label'] = 'Frequency (GHz)'
        line['y_label'] = 'S-parameter Magnitude'
        data_line = {'line': line, 'plot_type': 'one_line'}
        _, chosen_peaks = choose_peaks(data_line, load_results=False)

        prominence = chosen_peaks['current_peak']['prominence'] # in S-parameter units
        width = chosen_peaks['current_peak']['width'] # in GHz
        peak_search_params.append({'peak_freq': peak['freq'],
                                   'prominence': prominence,
                                   'width': width,
                                   'min_distance': min_distance})
    
    fields = data['field']

    found_modes = [{'freqs': [], 'fields': [], 'magnitudes': [], 'prominences': [], 'widths': []} for _ in range(len(peak_search_params))]

    print(f"\n{'='*60}")
    print(f"Starting peak tracking across {len(fields)} field values")
    print(f"Tracking {len(peak_search_params)} peaks")
    print(f"{'='*60}\n")

    for i, peak in enumerate(peak_search_params):
        for j, field in enumerate(fields):
            try:
                # Используем универсальную функцию поиска пика
                found_peak = useful_funcs.find_peak_flexible(
                    freqs=data['freq'],
                    s_values=np.abs(data['s_param'][j, :]),
                    expected_freq=peak['peak_freq'],
                    expected_width=peak['width'],
                    expected_prominence=peak['prominence'],
                    search_window=peak['min_distance']
                )
                
                print(f"[Peak {i+1}, Field {field:.1f}] Found by {found_peak['method']}: "
                      f"freq={found_peak['freq']:.4f} GHz, mag={found_peak['magnitude']:.4f}, "
                      f"prom={found_peak['prominence']:.4f}")
                
                # Сохраняем найденные значения
                found_modes[i]['freqs'].append(found_peak['freq'])
                found_modes[i]['fields'].append(field)
                found_modes[i]['magnitudes'].append(found_peak['magnitude'])
                found_modes[i]['prominences'].append(found_peak['prominence'])
                found_modes[i]['widths'].append(found_peak['width'])
                
                # Обновляем параметры поиска для следующего поля (трекинг пика)
                peak_search_params[i]['peak_freq'] = found_peak['freq']
                peak_search_params[i]['width'] = found_peak['width']
                peak_search_params[i]['prominence'] = found_peak['prominence'] * 0.8  # С запасом
                
            except ValueError as e:
                print(f"[Peak {i+1}, Field {field:.1f}] Error: {e}")
                continue

    output = {'peaks_data': {'modes': found_modes, 'plot_type': 'peaks', 'data': data}, 'plot_type': 'peaks'}

    return output, output

def approximate_peaks(data, load_results=False):
    if load_results:
        peaks_data, _ = pickle.load(open('selected_params.pkl', 'rb'))['peaks_data_fitted']
        return peaks_data, peaks_data
    modes = data['modes']
    full_data = data['data']

    fields = modes[0]['fields']

    for i, field in enumerate(fields):
        for j, mode in enumerate(modes):
            freqs = full_data['freq']
            field_idx = np.argmin(np.abs(full_data['field'] - field))
            s_values = np.abs(full_data['s_param'][field_idx, :])
            field_value = full_data['field'][field_idx]

            peak_freq = mode['freqs'][i]
            peak_width = mode['widths'][i]

            fit_region = (peak_freq - peak_width * 5, peak_freq + peak_width * 5)
            freq_indices = np.where((freqs >= fit_region[0]) & (freqs <= fit_region[1]))[0]

            local_freqs = freqs[freq_indices]
            local_s_values = s_values[freq_indices]
            amplitude_offset = np.min(local_s_values) if config_physics.PEAK_TYPE == 'maximum' else np.max(local_s_values)
            a = np.max(local_s_values) - amplitude_offset if config_physics.PEAK_TYPE == 'maximum' else amplitude_offset - np.min(local_s_values)

            try:
                popt, _ = sp.optimize.curve_fit(useful_funcs.fano_model,
                                                local_freqs,
                                                local_s_values,
                                                p0=[peak_freq, peak_width, 1.0, a, amplitude_offset])
                
                fitted_freq = popt[0]
                fitted_width = popt[1]
                fitted_q = popt[2]
                fitted_a = popt[3]
                fitted_offset = popt[4]

                plt.plot(local_freqs, local_s_values, 'b.', label='Data')
                fitted_curve = useful_funcs.fano_model(local_freqs, *popt)
                plt.plot(local_freqs, fitted_curve, 'r-', label='Fitted Curve')
                plt.legend()
                plt.show()
                time.sleep(0.5)
                plt.close()

                modes[j]['freqs'][i] = fitted_freq
                modes[j]['widths'][i] = fitted_width
                modes[j]['amplitude_offset'] = fitted_offset
                modes[j]['fano_q'] = fitted_q
                modes[j]['fano_a'] = fitted_a

                print(f"[Peak {j+1}, Field {field:.1f}] Fitted: freq={fitted_freq:.4f} GHz, width={fitted_width:.4f} GHz")

            except Exception as e:
                print(f"[Peak {j+1}, Field {field:.1f}] Fitting Error: {e}")
                continue
    output = {'peaks_data_fitted': {'modes': modes, 'data': full_data}, 'plot_type': 'peaks'}
    return output, output

def construct_own_modes(peaks):
    modes = [[] for _ in range(len(peaks))]

    for i, peak in enumerate(peaks):
        for point in peak:
            mode = point['freq'] + 1j * point['fwhm'] / 2
            modes[i].append(mode)
    
    return modes

def extract_coupling_params(resonator, own_modes):

    output = []

    fields = [point['field'] for point in peaks[0]]
    kappa = resonator['kappa']
    beta = resonator['beta']
    cavity_freq = resonator['resonance_freq']
    plato = resonator['plato']

    cavity_complex = cavity_freq - 1j * beta

    for i, field in enumerate(fields):
        mode_1 = own_modes[0][i]
        mode_2 = own_modes[1][i]

        delta = 2 * cavity_complex - (mode_1 + mode_2)
        coupling = np.sqrt((mode_1 - mode_2)**2 - delta**2) / 2

        J = coupling.real
        Gamma = -coupling.imag

        gamma = Gamma**2 * kappa

        magnon_complex = mode_1 + mode_2 - cavity_complex

        magnon_freq = magnon_complex.real
        alpha = - magnon_complex.imag

        params = {
            'field': field,
            'J': J,
            'Gamma': Gamma,
            'gamma': gamma,
            'magnon_freq': magnon_freq,
            'alpha': alpha
        }

        output.append(params)

    return output

def plot_results(data, peaks, resonator, own_modes, coupling_params):
    #График параметров резонатора
    plt.figure()
    plt.title('Resonator Parameters')
    fields = [point['field'] for point in peaks[0]]
    kappa_values = [resonator['kappa']] * len(fields)
    beta_values = [resonator['beta']] * len(fields)
    plt.plot(fields, kappa_values, label='Kappa')
    plt.plot(fields, beta_values, label='Beta')
    plt.xlabel('Field')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig('results/resonator_parameters.png')
    plt.show()

    #График собственных мод
    plt.figure()
    plt.title('Own Modes')

    modes = []
    for i, mode in enumerate(own_modes):
        mode_real = [m.real for m in mode]
        mode_imag = [m.imag for m in mode]
        modes.append({'mode_real': mode_real, 'mode_imag': mode_imag})
    
    plt.subplot(2, 1, 1)
    plt.title('Real Part of Own Modes')
    for i, mode in enumerate(modes):
        plt.plot(fields, mode['mode_real'], label=f'Mode {i+1} Real')
    plt.xlabel('Field')
    plt.ylabel('Frequency')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.title('Imaginary Part of Own Modes')
    for i, mode in enumerate(modes):
        plt.plot(fields, mode['mode_imag'], label=f'Mode {i+1} Imag')
    plt.xlabel('Field')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig('results/own_modes.png')
    plt.show()

    #График параметров связи и потерь магнонов
    plt.figure()
    plt.title('Coupling Parameters')
    J_values = [param['J'] for param in coupling_params]
    Gamma_values = [param['Gamma'] for param in coupling_params]
    gamma_values = [param['gamma'] for param in coupling_params]
    alpha_values = [param['alpha'] for param in coupling_params]

    plt.subplot(2, 1, 1)
    plt.title('Coupling Constants')
    plt.plot(fields, J_values, label='J')
    plt.plot(fields, Gamma_values, label='Gamma')
    plt.xlabel('Field')
    plt.ylabel('Value')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.title('Magnon Losses')
    plt.plot(fields, gamma_values, label='gamma')
    plt.plot(fields, alpha_values, label='alpha')
    plt.xlabel('Field')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig('results/coupling_parameters.png')
    plt.show()

def save_results(peaks, resonator, own_modes, coupling_params, filename):
    # Сохранение результатов в pkl файл
    results = {
        'peaks': peaks,
        'resonator': resonator,
        'own_modes': own_modes,
        'coupling_params': coupling_params
    }

    with open(filename, 'wb') as f:
        pickle.dump(results, f)

def reconstruct_own_modes(resonator, coupling_params, field):
    kappa = resonator['kappa']
    beta = resonator['beta']
    cavity_freq = resonator['resonance_freq']
    plato = resonator['plato']
    cavity_complex = cavity_freq - 1j * beta

    reconstructed_modes = {"mode_plus": [], "mode_minus": [], "fields": field}
    for param in coupling_params:
        J = param['J']
        Gamma = param['Gamma']

        magnon_complex = param['magnon_freq'] - 1j * param['alpha']
        coupling = J - 1j * Gamma
        mode_plus = (cavity_complex + magnon_complex) / 2 + np.sqrt(coupling**2 + ((cavity_complex - magnon_complex) / 2)**2)
        mode_minus = (cavity_complex + magnon_complex) / 2 - np.sqrt(coupling**2 + ((cavity_complex - magnon_complex) / 2)**2)

        reconstructed_modes["mode_plus"].append(mode_plus)
        reconstructed_modes["mode_minus"].append(mode_minus)   
    return reconstructed_modes

def reconstruct_data(resonator, coupling_params, freq, field):
    kappa = resonator['kappa']
    beta = resonator['beta']
    cavity_freq = resonator['resonance_freq']
    plato = resonator['plato']
    reconstructed_data = {"freq": freq, "field": field}

    J_matrix = np.array([[param['J']]*len(freq) for param in coupling_params])
    Gamma_matrix = np.array([[param['Gamma']]*len(freq) for param in coupling_params])
    gamma_matrix = np.array([[param['gamma']]*len(freq) for param in coupling_params])
    alpha_matrix = np.array([[param['alpha']]*len(freq) for param in coupling_params])
    magnon_freq_matrix = np.array([[param['magnon_freq']]*len(freq) for param in coupling_params])

    cavity_delta = freq - cavity_freq
    magnon_delta = freq - magnon_freq_matrix
    coupling = J_matrix - 1j * Gamma_matrix
    
    cavity_term = 1j * cavity_delta - (kappa + beta)
    magnon_term = 1j * magnon_delta - (gamma_matrix + alpha_matrix)

    response = plato + (kappa / (cavity_term - (coupling**2) / magnon_term))

    reconstructed_data["s_param"] = response

    return reconstructed_data

def visualize_approximated_modes(reconstructed_modes, own_modes):
    plt.figure()
    plt.title('Approximated Own Modes vs Original Own Modes')

    fields = reconstructed_modes['fields']
    mode_plus_real = [m.real for m in reconstructed_modes['mode_plus']]
    mode_minus_real = [m.real for m in reconstructed_modes['mode_minus']]
    own_mode_1_real = [mode.real for mode in own_modes[0]]
    own_mode_2_real = [mode.real for mode in own_modes[1]]

    plt.subplot(2, 1, 1)
    plt.title('Real Part Comparison')
    plt.plot(fields, mode_plus_real, label='Reconstructed Mode + Real')
    plt.plot(fields, mode_minus_real, label='Reconstructed Mode - Real')
    plt.plot(fields, own_mode_1_real, label='Original Mode 1 Real', linestyle='dashed')
    plt.plot(fields, own_mode_2_real, label='Original Mode 2 Real', linestyle='dashed')
    plt.xlabel('Field')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.title('Imaginary Part Comparison')
    mode_plus_imag = [m.imag for m in reconstructed_modes['mode_plus']]
    mode_minus_imag = [m.imag for m in reconstructed_modes['mode_minus']]
    own_mode_1_imag = [mode.imag for mode in own_modes[0]]
    own_mode_2_imag = [mode.imag for mode in own_modes[1]]

    plt.plot(fields, mode_plus_imag, label='Reconstructed Mode + Imag')
    plt.plot(fields, mode_minus_imag, label='Reconstructed Mode - Imag')
    plt.plot(fields, own_mode_1_imag, label='Original Mode 1 Imag', linestyle='dashed')
    plt.plot(fields, own_mode_2_imag, label='Original Mode 2 Imag', linestyle='dashed')
    plt.xlabel('Field')
    plt.legend()

    plt.savefig('results/approximated_own_modes.png')
    plt.show()

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


if __name__ == "__main__":
    data = data_io.load_s_parameter_data(config_data.FILEPATH)

    data = check_prosessing_results(data, preprocess_data, load_results=True)

    resonator = check_prosessing_results(data, extract_resonator_params, load_results=True)

    peaks = check_prosessing_results(data, find_peaks, load_results=True)

    peaks = check_prosessing_results(peaks, approximate_peaks, load_results=False)
    
    params = pickle.load(open('selected_params.pkl', 'rb'))

    print(params)


# peaks = find_peaks(data)

# own_modes = construct_own_modes(peaks)

# coupling_params = extract_coupling_params(resonator, own_modes)

# plot_results(data, peaks, resonator, own_modes, coupling_params)

# save_results(peaks, resonator, own_modes, coupling_params, 'results/analysis_output.pkl')

# reconstructed_modes = reconstruct_own_modes(resonator, coupling_params, data['field'])

# reconstructed_data = reconstruct_data(resonator, coupling_params, data['freq'], data['field'])

# visualize_approximated_modes(reconstructed_modes, own_modes)

# visualize_approximated_data(reconstructed_data, data)
