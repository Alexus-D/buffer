import numpy as np
import matplotlib.pyplot as plt

def choose_range(click, params, mode, ax):
    x_click, y_click = click
    output = {}
    clear_mode = False
    if params.get(mode) is None:
        output[mode] = [(x_click, y_click)]

        marker = ax.plot(x_click, y_click, 'ro')[0]
        ax.figure.canvas.draw()
        return clear_mode, output, marker
    elif len(params.get(mode)) < 2:
        y_min, y_max = np.min([params[mode][0][1], y_click]), np.max([params[mode][0][1], y_click])
        x_min, x_max = np.min([params[mode][0][0], x_click]), np.max([params[mode][0][0], x_click])
        output[mode] = {
            "x_range": (x_min, x_max),
            "y_range": (y_min, y_max)
        }
        clear_mode = True

        marker = ax.plot(x_click, y_click, 'ro')[0]
        ax.figure.canvas.draw()
        return clear_mode, output, marker
    else:
        raise ValueError("Both points have already been selected.")

def true_func(click, params, mode, ax):
    x_click, y_click = click
    clear_mode = True
    output = {mode: True}
    plt.close()
    return clear_mode, output, None

def false_func(click, params, mode, ax):
    x_click, y_click = click
    clear_mode = True
    output = {mode: False}
    plt.close()
    return clear_mode, output, None

def choose_point(click, params, mode, ax):
    x_click, y_click = click
    clear_mode = True
    output = {mode: (x_click, y_click)}

    marker = ax.plot(x_click, y_click, 'ro')[0]
    ax.figure.canvas.draw()
    return clear_mode, output, marker

def choose_points(click, params, mode, ax, num_points=2):
    x_click, y_click = click
    output = {}
    clear_mode = False
    if params.get(mode) is None:
        output[mode] = [(x_click, y_click)]

        marker = ax.plot(x_click, y_click, 'ro')[0]
        ax.figure.canvas.draw()
        return clear_mode, output, marker
    elif len(params.get(mode)) < num_points:
        output[mode] = params[mode]
        output[mode].append((x_click, y_click))
        if len(output[mode]) == num_points:
            clear_mode = True

        marker = ax.plot(x_click, y_click, 'ro')[0]
        ax.figure.canvas.draw()
        return clear_mode, output, marker
    else:
        raise ValueError("Both points have already been selected.")