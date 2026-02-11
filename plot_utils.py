import numpy as np
from numpy.fft import fft, ifft, fftshift, ifftshift
from scipy.signal import welch
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle, FancyArrow

from dataclasses import dataclass
from sigcom_toolkit.signal_utils import Signal_Utils
from sigcom_toolkit.general import GeneralConfig


@dataclass
class PlotUtilsConfig(GeneralConfig):
    pass


class Plot_Utils(Signal_Utils):
    def __init__(self, config: PlotUtilsConfig, **overrides):
        super().__init__(config, **overrides)


    def plot_signal(self, x=None, sigs=None, mode='time', scale='linear', plot_level=0, **kwargs):
        if self.plot_level<plot_level:
            return
        
        colors = ['blue', 'red', 'green', 'cyan', 'magenta', 'orange', 'purple']

        if isinstance(sigs, dict):
            sigs_dict = sigs
        else:
            sigs_dict = {"Signal": sigs}

        plt.figure()
        plot_args = kwargs.get('plot_args', {})

        for i, sig_name in enumerate(sigs_dict.keys()):
            if x is None:
                x = np.arange(len(sigs_dict[sig_name]))

            if mode=='time' or mode=='time_IQ':
                sig_plot = sigs_dict[sig_name].copy()
            elif mode=='fft':
                sig_plot = np.abs(fftshift(fft(sigs_dict[sig_name])))
            elif mode=='psd':
                fs = kwargs.get('fs', 1.0)
                nfft = kwargs.get('nfft', 2**(int(np.ceil(np.log2(len(x))))))
                freq, sig_plot = welch(sigs_dict[sig_name], fs, nperseg=nfft)
                x = freq
            
            if scale=='dB10':
                sig_plot = Signal_Utils.lin_to_db(sig_plot, mode='pow')
            if scale=='dB20':
                sig_plot = Signal_Utils.lin_to_db(sig_plot, mode='mag')
            elif scale=='linear':
                pass

            if mode!='time_IQ':
                plt.plot(x, sig_plot, color=colors[i], label=sig_name, **plot_args)
            else:
                plt.plot(x, np.real(sig_plot), color=colors[3*i], label='I', **plot_args)
                plt.plot(x, np.imag(sig_plot), color=colors[3*i+1], label='Q', **plot_args)
                plt.plot(x, np.abs(sig_plot), color=colors[3*i+2], label='Mag', **plot_args)

        title = kwargs.get('title', 'Signal in time domain')
        xlabel = kwargs.get('xlabel', 'Sample')
        ylabel = kwargs.get('ylabel', 'Magnitude (dB)')

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.minorticks_on()
        plt.grid(0.2)

        legend = kwargs.get('legend', False)
        if legend:
            plt.legend()

        plt.autoscale()
        if 'xlim' in kwargs:
            plt.xlim(kwargs['xlim'])
        if 'ylim' in kwargs:
            ylim=kwargs['ylim']
            if scale=='dB10':
                ylim = (Signal_Utils.lin_to_db(ylim[0], mode='pow'), Signal_Utils.lin_to_db(ylim[1], mode='pow'))
            if scale=='dB20':
                ylim = (Signal_Utils.lin_to_db(ylim[0], mode='mag'), Signal_Utils.lin_to_db(ylim[1], mode='mag'))
            plt.ylim(ylim)
        plt.tight_layout()

        # plt.axvline(x=30e6, color='g', linestyle='--', linewidth=1)

        plt.show()


    @staticmethod
    def set_plot_params(ax=None, lines=None, plot_params_dict=None):

        # Truncate the title to a maximum of 30 characters
        if plot_params_dict['title'] is not None:
            title = plot_params_dict['title']
            title = (title[:plot_params_dict['title_max_chars']] + '...') if len(title) > plot_params_dict['title_max_chars'] else title
            ax.set_title(plot_params_dict['title'])
        if plot_params_dict['xlabel'] is not None:
            ax.set_xlabel(plot_params_dict['xlabel'])
        if plot_params_dict['ylabel'] is not None:
            ax.set_ylabel(plot_params_dict['ylabel'])

        ax.title.set_fontsize(plot_params_dict['title_size'])
        ax.title.set_weight(plot_params_dict['title_weight'])
        ax.xaxis.label.set_fontsize(plot_params_dict['xaxis_size'])
        ax.yaxis.label.set_fontsize(plot_params_dict['yaxis_size'])
        ax.tick_params(axis='both', which='major', labelsize=plot_params_dict['ticks_size'])  # For major ticks
        ax.legend(fontsize=plot_params_dict['legend_size'])

        ax.grid(True)
        ax.minorticks_on()

        if lines is not None:
            for line in lines:
                line.set_linewidth(plot_params_dict['line_width'])

        plt.tight_layout()
        if plot_params_dict['hspace'] is not None and plot_params_dict['wspace'] is not None:
            plt.subplots_adjust(hspace=plot_params_dict['hspace'], wspace=plot_params_dict['wspace'])

        return ax, lines


    @staticmethod
    def draw_half_gauge(ax, min_val=-90, max_val=90):
        # Left half gauge
        ax.add_patch(Wedge((0.5, 0.5), 0.4, 90, -90, color="#B5A4D6", zorder=1))
        ax.add_patch(Wedge((0.5, 0.5), 0.35, 90, -90, color="#E6E6FA", zorder=2))
        # Right half gauge
        ax.add_patch(Wedge((0.5, 0.5), 0.4, -90, 90, color="#B5A4D6", zorder=1))
        ax.add_patch(Wedge((0.5, 0.5), 0.35, -90, 90, color="#E6E6FA", zorder=2))

        num_ticks = 18
        for i in range(num_ticks + 1):
            angle = i * (180 / num_ticks)
            tick_length = 0.05 if i % 2 == 0 else 0.03
            ax.plot([0.5 + 0.35 * np.cos(np.radians(angle)), 0.5 + (0.35 - tick_length) * np.cos(np.radians(angle))],
                    [0.5 + 0.35 * np.sin(np.radians(angle)), 0.5 + (0.35 - tick_length) * np.sin(np.radians(angle))],
                    color='black', lw=1, zorder=3)

        for i in range(num_ticks + 1):
            angle = i * (180 / num_ticks)
            value = -1 * (min_val + (max_val - min_val) * (i / num_ticks))
            x = 0.5 + 0.28 * np.cos(np.radians(angle))
            y = 0.5 + 0.28 * np.sin(np.radians(angle))
            ax.text(x, y, f'{int(value)}', fontsize=10, ha='center', va='center')

        ax.add_patch(Circle((0.5, 0.5), 0.05, color="black", zorder=5))
        ax.text(0.5, 0.95, "Angle of Arrival", fontsize=20, fontweight='bold', horizontalalignment='center')
        ax.set_aspect('equal')


    @staticmethod
    def gauge_update_needle(ax, value, min_val=90, max_val=-90):
        if value != np.nan and value != None:
            angle = (value - min_val) * 180 / (max_val - min_val)
        else:
            return
        x = 0.5 + 0.35 * np.cos(np.radians(angle))
        y = 0.5 + 0.35 * np.sin(np.radians(angle))

        arrow = FancyArrow(0.5, 0.5, x-0.5, y-0.5, width=0.02, head_width=0.05, head_length=0.08, color='#57068C', zorder=6)

        old_arrows = [p for p in ax.patches if isinstance(p, FancyArrow)]
        for old_arrow in old_arrows:
            old_arrow.remove()

        ax.add_patch(arrow)


