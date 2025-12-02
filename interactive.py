import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
import pickle

import models

class InteractiveParameterSelector2:
    def __init__(self, data, buttons, title='Interactive Parameter Selector', clear_buttons=True):
        self.mode = None

        self.data = data
        if isinstance(data, tuple):
            if isinstance(data[0], tuple):
                self.data = data[0][-1]
            else:
                self.data = data[0]

        self.params = {}

        self.button_labels = [button[0] for button in buttons]
        self.button_funcs = {button[1][0]: button[1][1] for button in buttons}
        self.button_modes = self.button_funcs.keys()
        self.markers = []

        if 'peaks' in ''.join(self.data.keys()):
            plot_type = 'peaks'
        else:
            plot_type = self.data['plot_type']

        if plot_type == 'contour':
            self._create_contour_plot__(title)
        elif plot_type == 'two_line':
            self._create_two_line_plot__(title)
        elif plot_type == 'one_line':
            self._create_one_line_plot__(title)
        elif plot_type == 'peaks':
            self._create_peaks_plot__(title)

        self._create_buttons__(clear_buttons)

        self.cid = None

    def _create_contour_plot__(self, title):
        self.fig = plt.figure(figsize=(12, 7))
        self.ax = self.fig.add_axes([0.25, 0.15, 0.7, 0.75])

        x = self.data['x']
        y = self.data['y']
        z = self.data['z']
        z = models.convert_linear_to_dB(z).transpose()

        x_label = self.data['x_label']
        y_label = self.data['y_label']
        z_label = self.data['z_label']

        contour = self.ax.contourf(x, y, z, levels=50, cmap='viridis')
        self.fig.colorbar(contour, ax=self.ax, label=z_label)
        self.ax.set_xlabel(x_label)
        self.ax.set_ylabel(y_label)
        self.ax.set_title(title)
    
    def _create_two_line_plot__(self, title):
        self.fig = plt.figure(figsize=(12, 7))
        self.ax = self.fig.add_axes([0.25, 0.15, 0.7, 0.75])

        line1 = self.data['line1']
        line2 = self.data['line2']
        x_label = line1.get('x_label', 'X-axis')
        y_label = line1.get('y_label', 'Y-axis')

        x1, y1, label1 = line1['x'], line1['y'], line1.get('label', 'Line 1')
        x2, y2, label2 = line2['x'], line2['y'], line2.get('label', 'Line 2')

        y1 = models.convert_linear_to_dB(y1)
        y2 = models.convert_linear_to_dB(y2)

        self.ax.plot(x1, y1, label=label1)
        self.ax.plot(x2, y2, label=label2)
        self.ax.set_xlabel(x_label)
        self.ax.set_ylabel(y_label)
        self.ax.set_title(title)
        self.ax.legend()
    
    def _create_one_line_plot__(self, title):
        self.fig = plt.figure(figsize=(12, 7))
        self.ax = self.fig.add_axes([0.25, 0.15, 0.7, 0.75])

        line = self.data['line']
        x_label = line.get('x_label', 'X-axis')
        y_label = line.get('y_label', 'Y-axis')

        x, y, label = line['x'], line['y'], line.get('label', 'Line')

        self.ax.plot(x, y, label=label)
        self.ax.set_xlabel(x_label)
        self.ax.set_ylabel(y_label)
        self.ax.set_title(title)
        self.ax.legend()

    def _create_peaks_plot__(self, title):
        self.fig = plt.figure(figsize=(12, 7))
        self.ax = self.fig.subplots(2, 1)

        self.ax[0].set_title(title)
        self.ax[0].set_xlabel('Field (Oe)')
        self.ax[0].set_ylabel('Frequency (GHz)')
        self.ax[1].set_xlabel('Field (Oe)')
        self.ax[1].set_ylabel('Width (MHz)')

        data_key = [i for i in self.data.keys() if 'peaks_data' in i][0]

        peak_1_values = self.data[data_key]['modes'][0]
        peak_2_values = self.data[data_key]['modes'][1]

        self.ax[0].plot(peak_1_values['fields'], peak_1_values['freqs'], 'ro-', label='Peak 1')
        self.ax[0].plot(peak_2_values['fields'], peak_2_values['freqs'], 'bo-', label='Peak 2')
        self.ax[0].legend()

        self.ax[1].plot(peak_1_values['fields'], peak_1_values['widths'], 'ro-', label='Peak 1 Width')
        self.ax[1].plot(peak_2_values['fields'], peak_2_values['widths'], 'bo-', label='Peak 2 Width')
        self.ax[1].legend()

    def _create_buttons__(self, clear_buttons):
        button_width = 0.15
        button_height = 0.04
        button_left = 0.02
        button_spacing = 0.06
        button_bottom = 0.15

        self.buttons = []  # Сохраняем ссылки на кнопки
        for i, mode in enumerate(self.button_funcs.keys()):
            ax_button = self.fig.add_axes([button_left,
                                           button_bottom + i * button_spacing,
                                           button_width,
                                           button_height])
            button = Button(ax_button, self.button_labels[i])
            button.on_clicked(lambda event, m=mode: self.__change_mode__(m))
            self.buttons.append(button)
        
        if clear_buttons:
            ax_btn_clear = self.fig.add_axes([button_left + button_width + 0.01, 0.05, 
                                            button_width, button_height])
            self.clear_button = Button(ax_btn_clear, 'Clear Params')
            self.clear_button.on_clicked(self.clear_params)
            self.buttons.append(self.clear_button)

            ax_btn_save = self.fig.add_axes([button_left, 0.05, 
                                            button_width, button_height])
            button = Button(ax_btn_save, "Save Params")
            button.on_clicked(self.save_params)
            self.buttons.append(button)

    def __change_mode__(self, mode):
        if self.cid is not None and mode is None:
            self.fig.canvas.mpl_disconnect(self.cid)
            self.cid = None
            self.mode = mode
        self.mode = mode
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.__on_click__)

    def __on_click__(self, event):           
        if self.mode in self.button_modes:
            field_click = event.xdata
            freq_click = event.ydata

            clear_mode, params, marker = self.button_funcs[self.mode]((field_click, freq_click), self.params, self.mode, self.ax)
            self.params.update(params)
            if marker is not None:
                self.markers.append(marker)
            if clear_mode:
                self.__change_mode__(None)
    
    def get_params(self):
        return self.params
    
    def clear_params(self, event):
        self.params = {}
        for marker in self.markers:
            marker.remove()
        self.markers = []
        self.ax.figure.canvas.draw()

    def save_params(self, event):
        plt.close()
        

def plot_interactive_contour_map(data, buttons, title='Interactive Contour Map', clear_buttons=True):
    selector = InteractiveParameterSelector2(data=data,
                                             buttons=buttons,
                                             title=title,
                                             clear_buttons=clear_buttons)

    plt.show()

    return selector
