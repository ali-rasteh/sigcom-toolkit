from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft, fftshift, ifft, ifftshift
from numpy.random import randn
from scipy.signal import firwin, freqz, lfilter, welch

from .general import General, GeneralConfig
from .plot_utils import Plot_Utils, PlotUtilsConfig


class AoAKalmanFilter:
    """
    Wrapped-angle Kalman filter with a persistent prior across windows.
    State is [angle; angular_rate]. All internal math uses radians; inputs/outputs
    to the public API are in degrees where noted.

    Parameters
    ----------
    dt : float
        Sampling period (in seconds) of the AoA measurements inside one fusion
        window (e.g., 0.1 for 100 ms). This also sets the discrete-time model step.
    sigma_meas_deg : float
        Standard deviation of the AoA measurement noise in degrees.
        Smaller -> the filter trusts measurements more; larger -> trusts the model more.
    sigma_acc_deg : float, optional (default=0.3)
        Standard deviation (in deg/s^2) of the angular acceleration driving
        the process noise. Larger -> more responsive to rapid changes (less smooth);
        smaller -> smoother output (more model-trusting).
    init_angle_deg : float or None, optional
        If provided, the filter is initialized with this AoA (in degrees).
        If None, the first observed sample in `step()` will be used to initialize.

    Notes
    -----
    - Angles are wrapped to (-pi, pi] internally to avoid discontinuities.
    - The constant-velocity (angle-rate) model is used:
        x_k = [ angle_k, angular_rate_k ]^T
        x_{k+1} = F x_k + w_k,  z_k = H x_k + v_k
      with F = [[1, dt], [0, 1]], H = [[1, 0]].
    """

    def __init__(self, dt, sigma_meas_deg, sigma_acc_deg=0.3, init_angle_deg=None):
        self.dt = float(dt)
        self.sigma_meas = float(np.deg2rad(sigma_meas_deg))
        self.sigma_acc = float(np.deg2rad(sigma_acc_deg))
        self.F = np.array([[1.0, self.dt], [0.0, 1.0]])
        self.H = np.array([[1.0, 0.0]])
        self.Q = self.sigma_acc**2 * np.array(
            [[self.dt**3 / 3.0, self.dt**2 / 2.0], [self.dt**2 / 2.0, self.dt]]
        )
        self.R = self.sigma_meas**2
        self.P = np.diag([np.deg2rad(30.0) ** 2, np.deg2rad(2.0) ** 2])
        self.initialized = False
        self.x = None
        if init_angle_deg is not None:
            self.reset(init_angle_deg)

    def wrap_angle_rad(self, a):
        return (a + np.pi) % (2 * np.pi) - np.pi

    def wrap_angle_deg(self, a):
        return (a + 180.0) % 360.0 - 180.0

    def reset(self, init_angle_deg):
        self.x = np.array([np.deg2rad(init_angle_deg), 0.0])
        self.x[0] = self.wrap_angle_rad(self.x[0])
        self.P = np.diag([np.deg2rad(30.0) ** 2, np.deg2rad(2.0) ** 2])
        self.initialized = True

    def step(self, angles_deg_1s):
        z_list = np.deg2rad(np.asarray(angles_deg_1s, dtype=float))
        if not self.initialized:
            self.reset(np.rad2deg(z_list[0]))
        for z in z_list:
            self.x = self.F @ self.x
            self.P = self.F @ self.P @ self.F.T + self.Q
            innov = self.wrap_angle_rad(z - (self.H @ self.x)[0])
            S = (self.H @ self.P @ self.H.T)[0, 0] + self.R
            K = (self.P @ self.H.T)[:, 0] / S
            self.x = self.x + K * innov
            self.P = (np.eye(2) - K[:, None] @ self.H) @ self.P
            self.x[0] = self.wrap_angle_rad(self.x[0])
        return float(np.rad2deg(self.x[0]))


@dataclass
class SignalUtilsConfig(GeneralConfig):
    fc: float = None
    fs: float = 1e9
    fs_tx: float = None
    fs_rx: float = None
    fs_trx: float = None
    n_samples: int = 1024
    n_samples_tx: int = None
    n_samples_rx: int = None
    n_samples_trx: int = None
    sc_range_ch: list = None
    n_samples_ch: int = None
    nfft: int = None
    nfft_tx: int = None
    nfft_rx: int = None
    nfft_trx: int = None
    nfft_ch: int = None

    ant_d: list = None
    wl: float = None
    steer_rad: list = None

    t: np.ndarray = None
    t_tx: np.ndarray = None
    t_rx: np.ndarray = None
    t_trx: np.ndarray = None
    t_ch: np.ndarray = None
    freq: np.ndarray = None
    freq_tx: np.ndarray = None
    freq_rx: np.ndarray = None
    freq_trx: np.ndarray = None
    freq_ch: np.ndarray = None
    om: np.ndarray = None
    om_tx: np.ndarray = None
    om_rx: np.ndarray = None
    om_trx: np.ndarray = None
    om_ch: np.ndarray = None

    def __post_init__(self):
        super().__post_init__()

        if self.fs_tx is None:
            self.fs_tx = self.fs
        if self.fs_rx is None:
            self.fs_rx = self.fs
        if self.fs_trx is None:
            self.fs_trx = self.fs
        if self.n_samples_tx is None:
            self.n_samples_tx = self.n_samples
        if self.n_samples_rx is None:
            self.n_samples_rx = self.n_samples
        if self.n_samples_trx is None:
            self.n_samples_trx = min(self.n_samples_tx, self.n_samples_rx)
        if self.sc_range_ch is None:
            self.sc_range_ch = [-1 * self.n_samples_trx // 2, self.n_samples_trx // 2 - 1]
        if self.n_samples_ch is None:
            self.n_samples_ch = self.sc_range_ch[1] - self.sc_range_ch[0] + 1
        if self.nfft is None:
            self.nfft = 2 ** np.ceil(np.log2(self.n_samples)).astype(int)
        if self.nfft_tx is None:
            self.nfft_tx = self.nfft
        if self.nfft_rx is None:
            self.nfft_rx = self.nfft
        if self.nfft_trx is None:
            self.nfft_trx = min(self.nfft_tx, self.nfft_rx)
        if self.nfft_ch is None:
            self.nfft_ch = self.n_samples_ch
        if self.t is None:
            self.t = np.arange(0, self.n_samples) / self.fs
        if self.t_tx is None:
            self.t_tx = np.arange(0, self.n_samples_tx) / self.fs_tx
        if self.t_rx is None:
            self.t_rx = np.arange(0, self.n_samples_rx) / self.fs_rx
        if self.t_trx is None:
            self.t_trx = np.arange(0, self.n_samples_trx) / self.fs_trx
        if self.t_ch is None:
            self.t_ch = np.arange(0, self.n_samples_ch) / self.fs_trx
        if self.freq is None:
            self.freq = np.linspace(-0.5, 0.5, self.nfft, endpoint=True) * self.fs / 1e6
        if self.freq_tx is None:
            self.freq_tx = np.linspace(-0.5, 0.5, self.nfft_tx, endpoint=True) * self.fs_tx / 1e6
        if self.freq_rx is None:
            self.freq_rx = np.linspace(-0.5, 0.5, self.nfft_rx, endpoint=True) * self.fs_rx / 1e6
        if self.freq_trx is None:
            self.freq_trx = np.linspace(-0.5, 0.5, self.nfft_trx, endpoint=True) * self.fs_trx / 1e6
        if self.freq_ch is None:
            self.freq_ch = (
                self.freq_trx[
                    (self.sc_range_ch[0] + self.nfft_trx // 2) : (
                        self.sc_range_ch[1] + self.nfft_trx // 2 + 1
                    )
                ]
                / 1e6
            )
        if self.om is None:
            self.om = np.linspace(-np.pi, np.pi, self.nfft, endpoint=True)
        if self.om_tx is None:
            self.om_tx = np.linspace(-np.pi, np.pi, self.nfft_tx, endpoint=True)
        if self.om_rx is None:
            self.om_rx = np.linspace(-np.pi, np.pi, self.nfft_rx, endpoint=True)
        if self.om_trx is None:
            self.om_trx = np.linspace(-np.pi, np.pi, self.nfft_trx, endpoint=True)
        if self.om_ch is None:
            self.om_ch = self.om_trx[
                (self.sc_range_ch[0] + self.nfft_trx // 2) : (
                    self.sc_range_ch[1] + self.nfft_trx // 2 + 1
                )
            ]


class Signal_Utils(General):
    def __init__(self, config: SignalUtilsConfig, **overrides):
        super().__init__(config, **overrides)

        plotter_config = PlotUtilsConfig().update_from_config(self.config)
        self.plotter = Plot_Utils(plotter_config)
        self.kalman_filter = AoAKalmanFilter(dt=0.1, sigma_meas_deg=np.sqrt(5.0), sigma_acc_deg=0.3)

    @staticmethod
    def lin_to_db(x, mode="pow"):
        if mode == "pow":
            return 10 * np.log10(x)
        elif mode == "mag":
            return 20 * np.log10(x)

    @staticmethod
    def db_to_lin(x, mode="pow"):
        if mode == "pow":
            return 10 ** (x / 10)
        elif mode == "mag":
            return 10 ** (x / 20)

    @staticmethod
    def wrap_angle(a, mode="rad"):
        if mode == "rad":
            # return (a + np.pi) % (2*np.pi) - np.pi
            return np.angle(np.exp(1j * a))
        elif mode == "deg":
            # return (a + 180.0) % 360.0 - 180.0
            return np.rad2deg(np.angle(np.exp(1j * np.deg2rad(a))))

    @staticmethod
    def aoa_to_phase(aoa, wl=0.01, ant_d_m=[0.0]):
        ant_dim = len(ant_d_m)
        if ant_dim == 1:
            phase = 2 * np.pi * ant_d_m[0] / wl * np.sin(aoa)
        elif ant_dim == 2:
            phase = 2 * np.pi * ant_d_m[0] / wl * np.sin(aoa[0]) + 2 * np.pi * ant_d_m[
                1
            ] / wl * np.sin(aoa[1])
        return phase

    @staticmethod
    def phase_to_aoa(phase, wl=0.01, ant_d_m=[0.0]):
        ant_dim = len(ant_d_m)
        if ant_dim == 1:
            aoa = np.arcsin(phase * wl / (2 * np.pi * ant_d_m[0]))
        elif ant_dim == 2:
            aoa = np.array(
                [
                    np.arcsin(phase * wl / (2 * np.pi * ant_d_m[0])),
                    np.arcsin(phase * wl / (2 * np.pi * ant_d_m[1])),
                ]
            )
        return aoa

    @staticmethod
    def mse(x, y):
        return np.mean(np.abs(x - y) ** 2)

    @staticmethod
    def sinc(x):
        sinc = np.sinc(x)  # sin(pi.x)/(pi.x)
        # sinc = np.sin(np.pi * x) / (np.pi * x)
        return sinc

    @staticmethod
    def rect(x):
        rect = np.where(np.abs(x) <= 0.5, 1.0, 0.0)
        return rect

    def plot_rect_sync(self):
        N = 1024  # Number of samples
        n = np.arange(-N / 2, N / 2)
        om = np.linspace(-np.pi, np.pi, N, endpoint=True)
        omega = np.pi / 16
        a = omega / np.pi
        M = 20

        sinc = self.sinc(a * n)
        rect = self.rect(n / M)
        self.plotter.plot_signal(n, {"sinc": np.abs(sinc)}, scale="linear", legend=True)
        self.plotter.plot_signal(
            om / np.pi,
            {"sinc_fft": np.abs(fftshift(fft(sinc))), "rect": self.rect(om / (2 * np.pi * a)) / a},
            scale="linear",
            legend=True,
        )
        self.plotter.plot_signal(n, {"rect": np.abs(rect)}, scale="linear", legend=True)
        self.plotter.plot_signal(
            om / np.pi,
            {
                "rect_fft": np.abs(fftshift(fft(rect))),
                "sinc": np.abs(self.sinc(om * (M + 1) / 2 / np.pi) * (M + 1)),
            },
            scale="linear",
            legend=True,
        )

    def dft(self, x):
        N = len(x)
        n = np.arange(N)
        k = n.reshape((N, 1))
        # Create the twiddle factor matrix (N x N)
        twiddle_factor = np.exp(-2j * np.pi * k * n / N)
        # Perform matrix multiplication
        X = np.dot(twiddle_factor, x)
        return X

    def psd(self, x, fs=None, nfft=None):
        if fs is None:
            fs = self.config.fs
        if nfft is None:
            nfft = self.config.nfft
        freq, psd = welch(x, fs, nperseg=nfft)
        return (freq, psd)

    def calculate_snr(self, sig_td, sig_sc_range=[0, 0]):
        # Calculate the SNR of a signal in the frequency domain
        sig_fd = fftshift(fft(sig_td, axis=-1))
        n_sig = sig_fd.shape[0]
        nfft = sig_fd.shape[-1]

        snrs = []
        # Calculate the power of the signal
        for i in range(n_sig):
            sig_fd_i = sig_fd[i, :]
            sig_and_noise = sig_fd_i[
                (sig_sc_range[0] + nfft // 2) : (sig_sc_range[1] + nfft // 2 + 1)
            ]
            sig_and_noise_power = np.mean(np.abs(sig_and_noise) ** 2)

            # Calculate the noise power
            noise_1 = sig_fd_i[: sig_sc_range[0] + nfft // 2]
            noise_2 = sig_fd_i[sig_sc_range[1] + nfft // 2 + 1 :]
            noise = np.concatenate((noise_1, noise_2), axis=-1)
            noise_power = np.mean(np.abs(noise) ** 2)

            # Calculate the SNR
            signal_power = sig_and_noise_power - noise_power
            snr = signal_power / noise_power
            snrs.append(snr)

        snrs = np.array(snrs)
        snr = np.mean(snrs)

        return snr

    def rotation_matrix(self, dim=2, angles=(0,)):
        if dim == 2:
            theta = angles[0]
            rotation_mat = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        elif dim == 3:
            # phi = angles[0]
            # theta = angles[1]
            raise NotImplementedError("3D rotation matrix not implemented yet")
        return rotation_mat

    def l2_norm(self, x):
        return np.linalg.norm(x)

    def upsample(self, signal, up=2):
        """
        Upsample a signal by a factor of 2 by inserting zeros between the original samples.

        Args:
            signal (np.array): Input signal to be upsampled.

        Returns:
            np.array: Upsampled signal with zeros inserted.
        """
        upsampled_len = up * len(signal)
        upsampled_sig = np.zeros(upsampled_len, dtype=complex)

        # Assign the original signal values to the even indices
        upsampled_sig[::up] = signal.copy()

        return upsampled_sig

    def cross_correlation(self, sig_1, sig_2, index):
        if index >= 0:
            padded_sig_2 = np.concatenate(
                (np.zeros(index, dtype=complex), sig_2[: len(sig_2) - index])
            )
        else:
            padded_sig_2 = np.concatenate((sig_2[-index:], np.zeros(-index, dtype=complex)))

        cros_corr = np.mean(sig_1 * np.conj(padded_sig_2))
        return cros_corr

    def integrate_signal(self, signal, n_samples=1024):
        n_ant = signal.shape[0]
        signal = signal.reshape(n_ant, -1, n_samples)
        signal = np.mean(signal, axis=1)

        return signal

    def extract_delay(self, sig_1, sig_2, plot_corr=False):
        """
        Calculate the delay of signal 1 with respect to signal 2 (signal 1 is ahead of signal 2)

        Args:
            sig_1 (np.array): First signal.
            sig_2 (np.array): Second signal.
            plot_corr (bool): Whether to plot the cross-correlation or not.

        Returns:
            delay (int): The delay of signal 1 with respect to signal 2 in samples.
        """
        cross_corr = np.correlate(sig_1, sig_2, mode="full")
        # cross_corr = np.correlate(sig_1, sig_2, mode='same')
        lags = np.arange(-len(sig_2) + 1, len(sig_1))

        if plot_corr:
            plt.figure()
            plt.plot(lags, np.abs(cross_corr), linewidth=1.0)
            plt.title("Cross-Correlation of the two signals")
            plt.xlabel("Lags")
            plt.ylabel("Correlation Coefficient")
            # plt.show()

        max_idx = np.argmax(np.abs(cross_corr))
        delay = int(lags[max_idx])
        # self.print(f'Time delay between the two signals: {delay} samples',4)
        return delay

    def extract_frac_delay(self, sig_1, sig_2, sc_range=[0, 0]):
        sig_1_f = fftshift(fft(sig_1, axis=-1))
        sig_2_f = fftshift(fft(sig_2, axis=-1))
        nfft = len(sig_1_f)

        phi = np.angle(sig_1_f * np.conj(sig_2_f))
        phi = phi[(sc_range[0] + nfft // 2) : (sc_range[1] + nfft // 2 + 1)]

        # Unwrap the phase to prevent discontinuities
        phi = np.unwrap(phi)

        # Perform linear regression to find the slope of the phase difference
        p = np.polyfit(np.arange(len(phi)), phi, deg=1)
        slope = p[0]  # Slope of the fitted line
        # Estimate the fractional delay using the slope
        frac_delay = -1 * (slope / (2 * np.pi)) * nfft

        return frac_delay

    @staticmethod
    def calc_phase_offset(sig_1, sig_2, sc_range=[0, 0]):
        # Return the phase offset between two signals in radians
        corr = np.correlate(sig_1, sig_2)
        max_idx = np.argmax(corr)
        phase_offest = np.angle(corr[max_idx])

        return phase_offest

    def adjust_phase(self, sig_1, sig_2, phase_offset):
        # Adjust the phase of sig_1 with respect to sig_2 based on the given phase offset
        sig_1_adj = sig_1 * np.exp(-1j * phase_offset)
        sig_2_adj = sig_2.copy()

        return sig_1_adj, sig_2_adj

    def time_adjust(self, sig_1, sig_2, delay):
        """
        Adjust the time of sig_1 with respect to sig_2 based on the given delay.

        Args:
            sig_1 (np.array): First signal.
            sig_2 (np.array): Second signal.
            delay (int): The delay of sig_1 with respect to sig_2 in samples.

        Returns:
            sig_1_adj (np.array): Adjusted sig_1.
            sig_2_adj (np.array): Adjusted sig_2.
            mse (float): Mean squared error between adjusted signals.
            err2sig_ratio (float): Ratio of MSE to mean squared value of sig_2.
        """
        n_points = min(sig_1.shape[0], sig_2.shape[0])
        delay = int(delay)

        sig_1_adj = np.roll(sig_1, -1 * delay)
        sig_2_adj = sig_2.copy()

        # mse = float(np.mean(np.abs(sig_1_adj[max(-1*delay,0):n_points+min(-1*delay,0)] - sig_2_adj[max(-1*delay,0):n_points+min(-1*delay,0)]) ** 2))
        mse = float(np.mean(np.abs(sig_1_adj[:n_points] - sig_2_adj[:n_points]) ** 2))
        err2sig_ratio = float(mse / np.mean(np.abs(sig_2) ** 2))

        return sig_1_adj, sig_2_adj, mse, err2sig_ratio

    def adjust_frac_delay(self, sig_1, sig_2, frac_delay):
        sig_1 = sig_1.copy()
        sig_2 = sig_2.copy()
        n_samples = sig_1.shape[0]

        sig_1_f = fftshift(fft(sig_1, axis=-1))
        sig_2_f = fftshift(fft(sig_2, axis=-1))
        omega = np.linspace(-np.pi, np.pi, n_samples)
        sig_1_f = np.exp(1j * omega * frac_delay) * sig_1_f
        sig_1_adj = ifft(ifftshift(sig_1_f), axis=-1)

        sig_2_adj = sig_2.copy()

        return sig_1_adj, sig_2_adj

    def gen_noise(self, mode="complex"):
        if mode == "real":
            noise = randn(self.config.n_samples).astype(
                complex
            )  # Generate noise with PSD=1/fs W/Hz
            # noise = normal(loc=0, scale=1, size=self.config.n_samples).astype(complex)
        elif mode == "complex":
            noise = (randn(self.config.n_samples) + 1j * randn(self.config.n_samples)).astype(
                complex
            )  # Generate noise with PSD=2/fs W/Hz

        return noise

    def slice_size(self, slice=None):
        if slice is None:
            size = 0
        else:
            size = 1
            for s in slice:
                size *= s.stop - s.start
        return size

    def slice_intersection(self, slice_1, slice_2):
        intersect = []
        if slice_1 is None or slice_2 is None:
            return None
        for s1, s2 in zip(slice_1, slice_2):
            start = max(s1.start, s2.start)
            stop = min(s1.stop, s2.stop)
            if start < stop:
                intersect.append(slice(start, stop))
            else:
                # If the slices do not intersect
                return None
        return tuple(intersect)

    def slice_union(self, slice_1, slice_2):
        union = []
        if slice_1 is None:
            return slice_2
        elif slice_2 is None:
            return slice_1
        for s1, s2 in zip(slice_1, slice_2):
            start = min(s1.start, s2.start)
            stop = max(s1.stop, s2.stop)
            if start < stop:
                union.append(slice(start, stop))
            else:
                # If the slices do not intersect
                return None
        return tuple(union)

    def compute_slices_similarity(self, predicted, target):
        if predicted is None and target is None:
            det_rate = None
            missed = None
            false_alarm = 0.0
            # det_rate = 1.0
            # missed = 0.0
            # false_alarm = 0.0
        elif predicted is not None and target is None:
            det_rate = None
            missed = None
            false_alarm = 1.0
            # det_rate = 0.0
            # missed = 0.0
            # false_alarm = 1.0
        elif predicted is None and target is not None:
            det_rate = None
            missed = 1.0
            false_alarm = None
            # det_rate = 0.0
            # missed = 1.0
            # false_alarm = 0.0
        else:
            intersection = self.slice_intersection(predicted, target)
            union = self.slice_union(predicted, target)
            intersection_size = self.slice_size(intersection)
            union_size = self.slice_size(union)

            # max_size = max(self.slice_size(slice_1), self.slice_size(slice_2))
            # det_rate = intersection_size / max_size
            det_rate = intersection_size / union_size
            missed = 0.0
            false_alarm = None
            # det_rate = intersection_size / union_size
            # missed = 0.0
            # false_alarm = 0.0

        return (det_rate, missed, false_alarm)

    def generate_tone(self, freq_mode="sc", sc=None, f=None, sig_mode="tone_2", gen_mode="fft"):
        if freq_mode == "sc":
            f = sc * self.config.fs_tx / self.config.nfft_tx
        elif freq_mode == "freq":
            sc = int(np.round((f) * self.config.nfft_tx / self.config.fs_tx))
        else:
            raise ValueError("Invalid frequency mode: " + freq_mode)

        if gen_mode == "time":
            wt = np.multiply(2 * np.pi * f, self.t_tx)
            if sig_mode == "tone_1":
                tone_td = np.cos(wt) + 1j * np.sin(wt)
            elif sig_mode == "tone_2":
                # tone_td = np.cos(wt) + 1j * np.cos(wt)
                tone_td = np.cos(wt)

        elif gen_mode == "fft":
            tone_fd = np.zeros((self.config.nfft_tx,), dtype="complex")
            if sig_mode == "tone_1":
                tone_fd[(self.config.nfft_tx >> 1) + sc] = 1
            elif sig_mode == "tone_2":
                tone_fd[(self.config.nfft_tx >> 1) + sc] = 1
                tone_fd[(self.config.nfft_tx >> 1) - sc] = 1
            tone_fd = fftshift(tone_fd, axes=0)

            # Convert the waveform to time-domain
            tone_td = np.fft.ifft(tone_fd, axis=0)

        # Normalize the signal
        tone_td /= np.max([np.abs(tone_td.real), np.abs(tone_td.imag)])

        self.print("Tone generation done", thr=2)

        return tone_td

    def generate_wideband(
        self,
        bw_mode="sc",
        sc_range=None,
        bw_range=None,
        wb_null_sc=0,
        modulation="4qam",
        sig_mode="wideband",
        gen_mode="fft",
        seed=100,
    ):
        if bw_mode == "sc":
            bw_range = [
                sc_range[0] * self.config.fs_tx / self.config.nfft_tx,
                sc_range[1] * self.config.fs_tx / self.config.nfft_tx,
            ]
        elif bw_mode == "freq":
            sc_range = [
                int(np.round(bw_range[0] * self.config.nfft_tx / self.config.fs_tx)),
                int(np.round(bw_range[1] * self.config.nfft_tx / self.config.fs_tx)),
            ]

        np.random.seed(seed)
        if gen_mode == "fft":
            if modulation == "psk":
                sym = [1, -1]
            elif modulation == "4qam":
                sym = [I + 1j * Q for I in [-1, 1] for Q in [-1, 1]]
            elif modulation == "16qam":
                sym = [I + 1j * Q for I in [-3, -1, 1, 3] for Q in [-3, -1, 1, 3]]
            elif modulation == "64qam":
                sym = [
                    I + 1j * Q
                    for I in [-7, -5, -3, -1, 1, 3, 5, 7]
                    for Q in [-7, -5, -3, -1, 1, 3, 5, 7]
                ]
            else:
                sym = []
                # raise ValueError('Invalid signal modulation: ' + modulation)

            # Create the wideband sequence in frequency-domain
            wb_fd = np.zeros((self.config.nfft_tx,), dtype="complex")
            if len(sym) > 0:
                wb_fd[
                    ((self.config.nfft_tx >> 1) + sc_range[0]) : (
                        (self.config.nfft_tx >> 1) + sc_range[1] + 1
                    )
                ] = np.random.choice(sym, len(range(sc_range[0], sc_range[1] + 1)))
            else:
                wb_fd[
                    ((self.config.nfft_tx >> 1) + sc_range[0]) : (
                        (self.config.nfft_tx >> 1) + sc_range[1] + 1
                    )
                ] = 1
            if sig_mode == "wideband_null":
                wb_fd[
                    ((self.config.nfft_tx >> 1) - wb_null_sc) : (
                        (self.config.nfft_tx >> 1) + wb_null_sc + 1
                    )
                ] = 0

            wb_fd = ifftshift(wb_fd, axes=0)

            # Convert the waveform to time-domain
            wb_td = ifft(wb_fd, axis=0)

        elif gen_mode == "ZadoffChu":
            prime_nums = [
                1,
                3,
                5,
                7,
                11,
                13,
                17,
            ]  # , 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
            cf = self.config.nfft_tx % 2
            q = 0
            # q = 0.5
            # u = np.random.choice(prime_nums)
            u = 3

            N = self.config.nfft_tx
            # N = self.config.sc_range[1] - self.config.sc_range[0] + 1
            n = np.arange(0, N)
            zc = np.exp(-1j * np.pi * u * n * (n + cf + 2 * q) / N)

            wb_fd = zc.copy()
            index_zeros = np.arange(
                self.config.sc_range[1], self.config.nfft_tx + self.config.sc_range[0]
            )
            wb_fd[index_zeros] = 0

            # wb_fd = ifftshift(wb_fd, axes=0)
            wb_td = ifft(wb_fd, axis=0)

        elif gen_mode == "ofdm":
            # N_blocks = 1000
            N_cp = 256
            N_fft = 768
            M = 16
            n_vec = np.arange(N_fft)
            x = np.exp(1j * np.pi * n_vec**2 / N_fft)
            x_cp = np.concatenate((x[-N_cp:], x))
            wb_td = x_cp
            # wb_td = np.tile(x_cp, N_blocks)

        # Normalize the signal
        wb_td /= np.max([np.abs(wb_td.real), np.abs(wb_td.imag)])

        self.print("Wide-band signal generation done", thr=2)

        return wb_td

    def create_mesh_grid(self, npoints=1000, xlim=[1, 1]):
        # Create a set of points x uniformly distributed in the area using meshgrid
        x1 = np.linspace(0, xlim[0], npoints)
        x2 = np.linspace(0, xlim[1], npoints)
        X1, X2 = np.meshgrid(x1, x2)
        X = np.zeros((npoints**2, 2))
        X[:, 0] = X1.flatten()
        X[:, 1] = X2.flatten()

    def beam_form(self, sigs):
        sigs_bf = sigs.copy()
        n_sigs = sigs.shape[0]
        ant_dim = len(self.config.ant_d)
        if ant_dim == 1:
            n_ant = n_sigs
        elif ant_dim == 2:
            n_ant_x = int(np.sqrt(n_sigs))
            n_ant_y = int(np.sqrt(n_sigs))

        for i in range(n_sigs):
            if ant_dim == 1:
                phase_shift = (
                    2 * np.pi * self.config.ant_d[0] * np.sin(self.config.steer_rad[0]) * i
                )
                print("phase_shift: ", phase_shift)
            elif ant_dim == 2:
                m = i // n_ant_y
                n = i % n_ant_y
                phase_shift = (
                    2
                    * np.pi
                    * (
                        m
                        * self.config.ant_d[0]
                        * np.sin(self.config.steer_rad[1])
                        * np.cos(self.config.steer_rad[0])
                        + n
                        * self.config.ant_d[1]
                        * np.sin(self.config.steer_rad[1])
                        * np.sin(self.config.steer_rad[0])
                    )
                )
            sigs_bf[i, :] = np.exp(1j * phase_shift) * sigs[i, :]

        return sigs_bf

    def filter(self, sig, center_freq=0, cutoff=50e6, fil_order=1000, plot=False):
        self.print("Starting to filter the signal...", thr=2)
        filter_fir = firwin(fil_order, cutoff / self.config.fs_rx)
        filter_fir = self.freq_shift(filter_fir, shift=center_freq, fs=self.config.fs_rx)

        if plot:
            plt.figure()
            w, h = freqz(filter_fir, worN=self.config.om_rx)
            plt.plot(w / np.pi, 20 * np.log10(np.abs(h)), linewidth=1.0)
            plt.title("Frequency response of the filter")
            plt.xlabel(r"Normalized Frequency ($\times \pi$ rad/sample)")
            plt.ylabel("Magnitude (dB)")
            plt.show()

        sig_fil = lfilter(filter_fir, 1, sig)
        # sig_fil = filtfilt(filter_fir, 1, sig)

        return sig_fil

    def freq_shift(self, sig, shift=0, fs=200e6):
        self.print("Shifting the signal in frequency domain...", thr=2)
        t = np.arange(0, len(sig)) / fs
        sig_shift = np.exp(2 * np.pi * 1j * shift * t) * sig

        return sig_shift

    def estimate_cfo(self, txtd, rxtd, mode="fine", sc_range=[0, 0]):
        n_samples = min(txtd.shape[1], rxtd.shape[1])
        txtd = txtd.copy()[:, :n_samples]
        rxtd = rxtd.copy()[:, :n_samples]

        # h_est_full = h_est_full.copy()
        txfd = fft(txtd, axis=-1)
        rxfd = fft(rxtd, axis=-1)

        n_rx_ant = rxtd.shape[0]
        n_tx_ant = txtd.shape[0]

        cfo_est = np.zeros((n_rx_ant))

        for tx_ant_id in range(1):
            for rx_ant_id in range(n_rx_ant):
                if mode == "coarse":
                    N = len(txtd[tx_ant_id])
                    # Compute the correlation between the two halves
                    Corr = np.sum(rxtd[rx_ant_id, : N // 2] * np.conj(rxtd[rx_ant_id, N // 2 : N]))
                    # Estimate the frequency offset
                    coarse_cfo = -1 * np.angle(Corr) / (2 * np.pi * (N // 2)) * (self.config.fs_rx)
                    cfo_est[rx_ant_id] = coarse_cfo

                elif mode == "fine":
                    # phi = np.angle(rxtd[rx_ant_id] * np.conj(txtd[tx_ant_id]))
                    phi = np.angle(rxfd[rx_ant_id] * np.conj(txfd[tx_ant_id]))
                    phi = phi[(sc_range[0] + n_samples // 2) : (sc_range[1] + n_samples // 2 + 1)]

                    # Unwrap the phase to prevent discontinuities
                    phi = np.unwrap(phi)

                    # Perform linear regression to find the slope of the phase difference
                    N = np.arange(len(phi))
                    p = np.polyfit(N, phi, deg=1)
                    slope = p[0]  # Slope of the fitted line
                    # Estimate the frequency offset using the slope
                    fine_cfo = (slope / (2 * np.pi)) * (self.config.fs_rx)
                    cfo_est[rx_ant_id] = fine_cfo

                else:
                    raise ValueError("Invalid CFO estimation mode: " + mode)

            # self.print(f"Estimated frequency offset: {} Hz".firmat(cfo_est), 0)

        return cfo_est

    def sync_frequency(self, rxtd, cfo, mode="time"):
        rxtd = rxtd.copy()
        n_rx_ant = rxtd.shape[0]
        rxfd = fft(rxtd, axis=-1)
        if mode == "time":
            for i in range(n_rx_ant):
                rxtd[i, :] = self.freq_shift(rxtd[i, :], shift=-1 * cfo[i], fs=self.config.fs_rx)
        elif mode == "freq":
            for i in range(n_rx_ant):
                rxfd[i, :] = self.freq_shift(rxfd[i, :], shift=-1 * cfo[i], fs=self.config.fs_rx)
            rxtd = ifft(rxfd, axis=-1)
        return rxtd

    def sync_time(self, rxtd, txtd, sc_range=[0, 0], rx_same_delay=False, sync_frac=False):
        n_samples_rx = rxtd.shape[-1]
        n_samples = min(txtd.shape[-1], rxtd.shape[-1])
        txtd_ = txtd.copy()[:, :n_samples]
        rxtd_ = rxtd.copy()[:, :n_samples]
        n_rx_ant = rxtd.shape[0]
        n_tx_ant = txtd.shape[0]
        rxtd_sync = np.zeros((n_rx_ant, n_tx_ant, n_samples_rx), dtype="complex")

        for tx_ant_id in range(n_tx_ant):
            for rx_ant_id in range(n_rx_ant):
                rx_id = 0 if rx_same_delay else rx_ant_id
                delay = self.extract_delay(rxtd_[rx_id], txtd_[tx_ant_id])
                rxtd_sync[rx_ant_id, tx_ant_id], _, _, _ = self.time_adjust(
                    rxtd[rx_ant_id], txtd_[tx_ant_id], delay
                )

                if sync_frac:
                    frac_delay = self.extract_frac_delay(
                        rxtd_sync[rx_ant_id, tx_ant_id, :n_samples],
                        txtd_[tx_ant_id],
                        sc_range=sc_range,
                    )
                    # print(f"Fractional delay: {frac_delay}")
                    rxtd_sync[rx_ant_id, tx_ant_id], _ = self.adjust_frac_delay(
                        rxtd_sync[rx_ant_id, tx_ant_id], txtd_[tx_ant_id], frac_delay
                    )

        return rxtd_sync

    def sparse_est(
        self,
        h,
        g=None,
        sc_range_ch=[0, 0],
        npaths=[1, 1],
        nframe_avg=1,
        ndly=10000,
        drange=[-6, 20],
        cv=False,
        n_ignore=-1,
    ):
        """
        Estimates the sparse channel using Orthogonal Matching Pursuit (OMP).
        Parameters:
        -----------
        h : np.array of shape (nfft, nrx, ntx, nframe)
            The time-domain channel estimate.
        g : np.array of shape (nfft, nrx, ntx)
            The system response in the time-domain.
        npaths : list of ints, optional
            Maximum number of paths to estimate in each round. Default is [1,1].
        nframe_avg : int, optional
            Number of frames to average for channel estimation. Default is 1.
        ndly : int, optional
            Number of delay points to test around the peak. Default is 10000.
        drange : list, optional
            Range of sample delays to test around the peak. Default is [-6, 20].
        cv : bool, optional
            Whether to use cross-validation to stop the path estimation. Default is True.
        Raises:
        -------
        ValueError
            If there are not enough frames for cross-validation or averaging.
        Notes:
        ------
        - The method uses cross-validation to stop the path estimation when the test error exceeds the training error by a certain tolerance.
        - The delays are set to test around the peak of the time-domain channel estimate.
        - The method uses Orthogonal Matching Pursuit (OMP) to find the sparse solution.
        """

        # Number of paths stops when test error exceeds training error
        # by 1+cv_tol
        cv_tol = 0.1

        # Compute the channel estimates for training and test
        # by averaging over the different frames

        H = fft(h, axis=0)
        nframe = H.shape[3]
        nfft = H.shape[0]
        n_rx_ant = H.shape[1]
        n_tx_ant = H.shape[2]
        if g is None:
            G = np.ones((H.shape[0], H.shape[1], H.shape[2]), dtype="complex")
            g = ifft(G, axis=0)
            nff_g = nfft
        else:
            G = fft(g, axis=0)
            nff_g = G.shape[0]
        G = ifftshift(
            fftshift(G, axes=0)[(sc_range_ch[0] + nff_g // 2) : (sc_range_ch[1] + nff_g // 2 + 1)],
            axes=0,
        )

        h_tr_mat = [[None for i in range(n_tx_ant)] for j in range(n_rx_ant)]
        dly_est_mat = [[None for i in range(n_tx_ant)] for j in range(n_rx_ant)]
        peaks_mat = [[None for i in range(n_tx_ant)] for j in range(n_rx_ant)]
        npaths_est_mat = [[None for i in range(n_tx_ant)] for j in range(n_rx_ant)]

        for irx in range(n_rx_ant):
            for itx in range(n_tx_ant):
                if cv:
                    if nframe < 2 * nframe_avg:
                        raise ValueError("Not enough frames for cross-validation")
                    Itr = np.arange(0, nframe_avg) * 2
                    Its = Itr + 1
                    H_tr = np.mean(H[:, irx, itx, Itr], axis=1)
                    H_ts = np.mean(H[:, irx, itx, Its], axis=1)

                    # For the FA probability, we set the threhold to the energy
                    # of the max on nfft random basis functions.  The energy
                    # on each basis function is exponential with mean 1/nfft.
                    # So, the maximum energy is exponential with mean 1/nfft* (\sum_k 1/k)
                    t = np.arange(1, nfft)
                    cv_dec = 1 - 2 * np.sum(1 / t) / nfft
                else:
                    if nframe < nframe_avg:
                        raise ValueError("Not enough frames for averaging")
                    H_tr = H[:, irx, itx, :nframe_avg]
                    H_tr = np.mean(H_tr, axis=1)
                h_tr = np.fft.ifft(H_tr, axis=0)

                # Set the delays to test around the peak
                idx = np.argmax(np.abs(h_tr))

                dly_test = (idx + np.linspace(drange[0], drange[1], ndly)) / self.config.fs_trx
                # Create the basis vectors
                freq = (
                    (np.arange(nfft) / nfft) * self.config.fs_trx
                    + self.config.fc
                    - self.config.fs_trx / 2
                )
                B = G[:, irx, itx, None] * np.exp(
                    -2 * np.pi * 1j * freq[:, None] * dly_test[None, :]
                )

                # Use OMP to find the sparse solution
                coeff_est = np.zeros(npaths[0])

                resid = H_tr.copy()
                indices = []
                indices1 = []
                mse_tr = np.zeros(npaths[0])
                mse_ts = np.zeros(npaths[0])

                npaths_est = 0
                for i in range(npaths[0]):
                    # Compute the correlation
                    cor = np.abs(B.conj().T.dot(resid))

                    # Add the highest correlation to the list
                    idx = np.argmax(cor)
                    indices1.append(idx)

                    # Use least squares to estimate the coefficients
                    coeffs_est = np.linalg.lstsq(B[:, indices1], H_tr, rcond=None)[0]

                    # Compute the resulting sparse channel
                    H_sparse = B[:, indices1].dot(coeffs_est)

                    # Compute the current residual
                    resid = H_tr - H_sparse

                    # Compute the MSE on the training data
                    mse_tr[i] = np.mean(np.abs(resid) ** 2) / np.mean(np.abs(H_tr) ** 2)

                    # Compute the MSE on the test data if CV is used
                    if cv:
                        resid_ts = H_ts - H_sparse
                        mse_ts[i] = np.mean(np.abs(resid_ts) ** 2) / np.mean(np.abs(H_ts) ** 2)

                        # Check if path is valid
                        if i > 0:
                            if mse_ts[i] > cv_dec * mse_ts[i - 1]:
                                break
                        if mse_ts[i] > (1 + cv_tol) * mse_tr[i]:
                            break

                    # Updated the number of paths
                    npaths_est = i + 1
                    indices.append(idx)

                # Ignore the paths that are too close to the first path
                n_ignore_ = n_ignore * (ndly // (drange[1] - drange[0]))
                indices1 = indices.copy()
                for index in indices1[1:]:
                    if index <= indices[0] + n_ignore_ and index >= indices[0] - n_ignore_:
                        indices.remove(index)
                indices = indices[: npaths[1]]
                npaths_est = len(indices)

                dly_est = dly_test[indices]
                dly_est = np.pad(dly_est, (0, npaths[1] - npaths_est), constant_values=0)

                # Use least squares to estimate the coefficients
                coeffs_est = np.linalg.lstsq(B[:, indices], H_tr, rcond=None)[0]

                # Compute the resulting sparse channel
                H_sparse = B[:, indices].dot(coeffs_est)
                h_sparse = np.fft.ifft(H_sparse, axis=0)

                scale = np.mean(np.abs(G)) ** 2
                # peaks  = np.abs(coeffs_est)**2 * scale
                peaks = coeffs_est.copy() * np.sqrt(scale)
                peaks = np.pad(peaks, (0, npaths[1] - npaths_est), constant_values=0)

                h_tr_mat[irx][itx] = h_tr.copy()
                if len(dly_est) == npaths[1]:
                    dly_est_mat[irx][itx] = dly_est.copy()
                else:
                    # dly_est_mat[irx][itx] = dly_est.copy().extend([0]*(npaths-npaths_est))
                    dly_est_mat[irx][itx] = np.array([0] * npaths[1])
                if len(peaks) == npaths[1]:
                    peaks_mat[irx][itx] = peaks.copy()
                else:
                    # peaks_mat[irx][itx] = peaks.copy() + [0]*(npaths-npaths_est)
                    peaks_mat[irx][itx] = np.array([0] * npaths[1])

                npaths_est_mat[irx][itx] = npaths_est

        h_tr_mat = np.array(h_tr_mat)
        dly_est_mat = np.array(dly_est_mat)
        peaks_mat = np.array(peaks_mat)
        npaths_est_mat = np.array(npaths_est_mat)

        return (h_tr_mat, dly_est_mat, peaks_mat, npaths_est_mat)

    def channel_estimate(self, txtd, rxtd_s, sys_response=None, sc_range_ch=[0, 0], snr_est=100):
        if len(rxtd_s.shape) == 4:
            rxtd_s = np.mean(rxtd_s.copy(), axis=0)
        deconv_sys_response = sys_response is not None

        n_samples = min(txtd.shape[-1], rxtd_s.shape[-1])
        n_samples_ch = sc_range_ch[1] - sc_range_ch[0] + 1
        # n_samples_ch = n_samples

        txtd = txtd.copy()[:, :n_samples]
        rxtd_s = rxtd_s.copy()[:, :, :n_samples]
        n_rx_ant = rxtd_s.shape[0]
        n_tx_ant = txtd.shape[0]

        t_ch = self.config.t_trx[:n_samples_ch]
        freq_ch = self.config.freq_trx[
            (sc_range_ch[0] + n_samples // 2) : (sc_range_ch[1] + n_samples // 2 + 1)
        ]

        H_est_full = np.zeros((n_rx_ant, n_tx_ant, n_samples_ch), dtype="complex")
        h_est_full = np.zeros((n_rx_ant, n_tx_ant, n_samples_ch), dtype="complex")

        txfd = fft(txtd, axis=-1)
        rxfd_s = fft(rxtd_s, axis=-1)
        # rxfd_s = np.roll(rxfd_s, 1, axis=1)
        # txfd = np.roll(txfd, 1, axis=1)

        if deconv_sys_response:
            g = sys_response.copy()[:, :n_samples]
            G = fft(g, axis=-1)

        for tx_ant_id in range(n_tx_ant):
            for rx_ant_id in range(n_rx_ant):
                if deconv_sys_response:
                    txfd_ = txfd[tx_ant_id] * G[rx_ant_id, tx_ant_id]
                else:
                    txfd_ = txfd[tx_ant_id]
                rxfd_ = rxfd_s[rx_ant_id, tx_ant_id]
                rx_pow = np.mean(np.abs(rxfd_) ** 2)
                noise_pow = rx_pow / snr_est
                H_est_full_ = (
                    rxfd_s[rx_ant_id, tx_ant_id]
                    * np.conj(txfd_)
                    / ((np.abs(txfd_) ** 2) + noise_pow)
                )
                # H_est_full_ = rxfd[rx_ant_id] * np.conj(txfd_)
                # H_est_full_ = rxfd[rx_ant_id] / txfd_

                H_est_full_ = ifftshift(
                    fftshift(H_est_full_)[
                        (sc_range_ch[0] + n_samples // 2) : (sc_range_ch[1] + n_samples // 2 + 1)
                    ]
                )

                h_est_full_ = ifft(H_est_full_)
                H_est_full[rx_ant_id, tx_ant_id, :] = H_est_full_.copy()
                h_est_full[rx_ant_id, tx_ant_id, :] = h_est_full_.copy()

                im = np.argmax(np.abs(h_est_full_))
                h_est_full_ = np.roll(h_est_full_, -im + len(h_est_full_) // 10)
                h_est_full_ = h_est_full_.flatten()

                sig = np.abs(h_est_full_) / np.max(np.abs(h_est_full_))
                title = "Channel response in the time domain \n between TX antenna {} and RX antenna {}".format(
                    tx_ant_id, rx_ant_id
                )
                xlabel = "Time (s)"
                ylabel = "Normalized Magnitude (dB)"
                self.plotter.plot_signal(
                    t_ch, sig, scale="dB20", title=title, xlabel=xlabel, ylabel=ylabel, plot_level=5
                )

                sig = np.abs(fftshift(H_est_full_))
                title = "Channel response in the frequency domain \n between TX antenna {} and RX antenna {}".format(
                    tx_ant_id, rx_ant_id
                )
                xlabel = "Frequency (MHz)"
                ylabel = "Magnitude (dB)"
                self.plotter.plot_signal(
                    freq_ch,
                    sig,
                    scale="dB20",
                    title=title,
                    xlabel=xlabel,
                    ylabel=ylabel,
                    plot_level=5,
                )

        # H_est = np.linalg.pinv(txfd.T) @ rxfd.T
        # H_est = H_est.T
        # H_est = rxfd @ np.linalg.pinv(txfd)
        H_est = np.mean(H_est_full, axis=-1)

        time_pow = np.sum(np.abs(H_est_full) ** 2, axis=(0, 1))
        idx_max = np.argmax(time_pow)
        H_est_max = H_est_full[:, :, idx_max]

        return h_est_full, H_est, H_est_max

    def channel_equalize(
        self,
        txtd,
        rxtd,
        h_full,
        H,
        sc_range=[0, 0],
        sc_range_ch=[0, 0],
        null_sc_range=[0, 0],
        n_rx_ch_eq=1,
    ):
        n_samples = min(txtd.shape[-1], rxtd.shape[-1])
        n_samples_ch = sc_range_ch[1] - sc_range_ch[0] + 1
        txtd = txtd.copy()[:, :n_samples]
        rxtd = rxtd.copy()[:, :n_samples]

        txfd = fft(txtd, axis=-1)
        rxfd = fft(rxtd, axis=-1)
        H_full = fft(h_full, axis=-1)

        rxtd_eq = rxtd.copy()
        rxfd_eq = rxfd.copy()

        rxfd_ = fftshift(rxfd, axes=-1)
        rxfd_eq_ = fftshift(rxfd_eq, axes=-1)
        H_full_ = fftshift(H_full, axes=-1)
        if n_rx_ch_eq == 1:
            for rx_ant_id in range(rxtd_eq.shape[0]):
                # rxfd_eq[rx_ant_id] = rxfd[rx_ant_id] / H[rx_ant_id, rx_ant_id]

                for i, sc in enumerate(range(sc_range[0], sc_range[1] + 1)):
                    if sc not in range(null_sc_range[0], null_sc_range[1] + 1):
                        rxfd_eq_[rx_ant_id, sc + n_samples // 2] = (
                            rxfd_[rx_ant_id, sc + n_samples // 2] / H_full_[rx_ant_id, rx_ant_id, i]
                        )
                rxfd_eq[rx_ant_id] = ifftshift(rxfd_eq_[rx_ant_id])
        else:
            tol = 1e-6
            for i, sc in enumerate(range(sc_range[0], sc_range[1] + 1)):
                if sc not in range(null_sc_range[0], null_sc_range[1] + 1):
                    H_sc = H_full_[:, :, i]
                    # H_sc += tol*np.eye(H_sc.shape[0])
                    # H_sc_inv = np.linalg.pinv(H_sc)
                    H_sc_inv = (np.conj(H_sc.T) * H_sc + tol * np.eye(H_sc.shape[0])) * np.conj(
                        H_sc.T
                    )
                    rxfd_eq_[:, sc + n_samples // 2] = H_sc_inv @ rxfd_[:, sc + n_samples // 2]

            rxfd_eq = ifftshift(rxfd_eq_, axes=-1)

        rxtd_eq = ifft(rxfd_eq, axis=-1)

        return rxtd_eq

    def filter_aoa(self, rx_phase_list, rx_phase, aoa_list, aoa):
        # alpha_phase = 0.5
        # alpha_aoa = 0.5
        alpha_phase = 1.0
        alpha_aoa = 1.0

        if len(aoa_list) > 0:
            aoa_last = aoa_list[-1]
        else:
            if aoa is None:
                aoa_last = 0
            else:
                aoa_last = aoa
        if aoa is None:
            aoa = aoa_last
        else:
            # aoa = alpha_aoa * aoa + (1 - alpha_aoa) * aoa_last
            # Use Kalman filter to smooth the AOA gauge signal
            if not aoa_list:
                aoa_list.append(aoa)
            window_deg = np.rad2deg(aoa_list[-10:])
            aoa = self.wrap_angle(self.kalman_filter.step(window_deg), mode="deg")
            aoa = np.deg2rad(aoa)

        if len(rx_phase_list) > 0:
            rx_phase_last = rx_phase_list[-1]
        else:
            if rx_phase is None:
                rx_phase_last = 0
            else:
                rx_phase_last = rx_phase
        if rx_phase is None:
            rx_phase = rx_phase_last
        else:
            # rx_phase = alpha_phase * rx_phase + (1 - alpha_phase) * rx_phase_last
            # Use Kalman filter to smooth the RX phase signal
            if not rx_phase_list:
                rx_phase_list.append(rx_phase)
            window_deg = np.rad2deg(rx_phase_list[-10:])
            rx_phase = self.wrap_angle(self.kalman_filter.step(window_deg), mode="deg")
            rx_phase = np.deg2rad(rx_phase)

        rx_phase_list.append(rx_phase)
        aoa_list.append(aoa)

        return rx_phase_list, aoa_list

    def angle_of_arrival(
        self,
        txtd,
        rxtd,
        h_full,
        rx_phase_list,
        aoa_list,
        fc=1e9,
        rx_phase_offset=0,
        rx_delay_offset=0,
    ):
        if len(rxtd.shape) == 3:
            rxtd = np.mean(rxtd.copy(), axis=0)
        rx_phase = Signal_Utils.calc_phase_offset(rxtd[0, :], rxtd[1, :])

        rx_phase = np.angle(np.exp(1j * rx_phase))
        rx_phase -= rx_phase_offset
        # rx_phase -= (rx_delay_offset * 2 * np.pi * fc)

        # Wrap phase between -pi and pi
        rx_phase = np.angle(np.exp(1j * rx_phase))

        angle_sin = rx_phase / (2 * np.pi * self.config.ant_d[0])
        if angle_sin > 1 or angle_sin < -1:
            # angle = np.nan
            aoa = None
            rx_phase = None
            self.print("AoA sin is out of range: {}".format(angle_sin), 1)
        else:
            aoa = np.arcsin(angle_sin)

        rx_phase_list, aoa_list = self.filter_aoa(rx_phase_list, rx_phase, aoa_list, aoa)

        return rx_phase_list, aoa_list

    def estimate_mimo_params(self, txtd, rxtd, fc, h_full, H, rx_phase_list, aoa_list):
        # U, S, Vh = np.linalg.svd(H)
        # W_tx = Vh.conj().T
        # W_rx = U
        # rx_phase = np.mean(np.angle(U[0,:]*np.conj(U[1,:])))
        # tx_phase = np.mean(np.angle(Vh[:,0]*np.conj(Vh[:,1])))

        rx_phase_list, aoa_list = self.angle_of_arrival(
            txtd=txtd,
            rxtd=rxtd,
            h_full=h_full,
            rx_phase_list=rx_phase_list,
            aoa_list=aoa_list,
            fc=fc,
            rx_phase_offset=self.rx_phase_offset,
            rx_delay_offset=self.rx_delay_offset,
        )
        # print("AoA: {} deg".format(np.rad2deg(aoa)))

        return rx_phase_list, aoa_list
