import numpy as np
import matplotlib.pyplot as plt

def choose_range(click, params, mode, ax):
    field_click, freq_click = click
    output = {}
    clear_mode = False
    if params.get(mode) is None:
        output[mode] = [(field_click, freq_click)]

        marker = ax.plot(field_click, freq_click, 'ro')[0]
        ax.figure.canvas.draw()
        return clear_mode, output, marker
    elif len(params.get(mode)) < 2:
        freq_min, freq_max = np.min([params[mode][0][1], freq_click]), np.max([params[mode][0][1], freq_click])
        field_min, field_max = np.min([params[mode][0][0], field_click]), np.max([params[mode][0][0], field_click])
        output[mode] = {
            "field_range": (field_min, field_max),
            "freq_range": (freq_min, freq_max)
        }
        clear_mode = True

        marker = ax.plot(field_click, freq_click, 'ro')[0]
        ax.figure.canvas.draw()
        return clear_mode, output, marker
    else:
        raise ValueError("Both points have already been selected.")

def true_func(click, params, mode, ax):
    field_click, freq_click = click
    clear_mode = True
    output = {mode: True}
    plt.close()
    return clear_mode, output, None

def false_func(click, params, mode, ax):
    field_click, freq_click = click
    clear_mode = True
    output = {mode: False}
    plt.close()
    return clear_mode, output, None

def choose_point(click, params, mode, ax):
    field_click, freq_click = click
    clear_mode = True
    output = {mode: (field_click, freq_click)}

    marker = ax.plot(field_click, freq_click, 'ro')[0]
    ax.figure.canvas.draw()
    return clear_mode, output, marker

def choose_points(click, params, mode, ax, num_points=2):
    field_click, freq_click = click
    output = {}
    clear_mode = False
    if params.get(mode) is None:
        output[mode] = [(field_click, freq_click)]

        marker = ax.plot(field_click, freq_click, 'ro')[0]
        ax.figure.canvas.draw()
        return clear_mode, output, marker
    elif len(params.get(mode)) < num_points:
        output[mode] = params[mode]
        output[mode].append((field_click, freq_click))
        if len(output[mode]) == num_points:
            clear_mode = True

        marker = ax.plot(field_click, freq_click, 'ro')[0]
        ax.figure.canvas.draw()
        return clear_mode, output, marker
    else:
        raise ValueError("Both points have already been selected.")