import contextlib
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft, fftshift
from numpy.random import exponential, rand, randint, uniform
from scipy.signal import firwin, lfilter

from .signal_utils import SignalUtils, SignalUtilsConfig

with contextlib.suppress(BaseException):
    import torch   # type: ignore # noqa: I001


@dataclass
class SpecSenseUtilsConfig(SignalUtilsConfig):
    n_sigs_max: int = None
    size_sam_mode: str = None
    snr_sam_mode: str = None
    noise_power: float = None
    mask_mode: str = None


class SpecSenseUtils(SignalUtils):
    def __init__(self, config: SpecSenseUtilsConfig, **overrides):
        super().__init__(config, **overrides)

    @staticmethod
    def generate_random_regions(
        shape=(1000,), n_regions=1, min_size=None, max_size=None, size_sam_mode="log"
    ):
        regions = []
        # ndims = len(shape)
        for _ in range(n_regions):
            region_slices = []
            for d, dim in enumerate(shape):
                if min_size is not None and max_size is not None:
                    s1 = min_size[d]
                    s2 = max_size[d] + 1
                else:
                    s1 = 1
                    s2 = min(101, (dim + 1) // 2 + 1)
                if size_sam_mode == "lin":
                    size = randint(s1, s2)
                elif size_sam_mode == "log":
                    margin = 1e-9
                    size = uniform(np.log10(s1), np.log10(s2 - margin))
                    size = int(10**size)
                start = randint(0, dim - size + 1)
                size = min(size, dim - start)
                region_slices.append(slice(start, start + size))
            regions.append(tuple(region_slices))

        return regions

    def generate_random_psd(
        self,
        shape=(1000,),
        sig_regions=None,
        n_sigs=1,
        n_sigs_max=1,
        sig_size_min=None,
        sig_size_max=None,
        noise_power=1,
        snr_range=np.array([10, 10]),
        size_sam_mode="log",
        snr_sam_mode="log",
        mask_mode="binary",
    ):

        sig_power_range = noise_power * snr_range.astype(float)
        psd = exponential(noise_power, shape)
        if mask_mode == "binary" or mask_mode == "snr":
            mask = np.zeros(shape, dtype=float)
        elif mask_mode == "channels":
            mask = np.zeros((n_sigs_max,) + shape, dtype=float)

        if sig_regions is None:
            regions = self.generate_random_regions(
                shape=shape,
                n_regions=n_sigs,
                min_size=sig_size_min,
                max_size=sig_size_max,
                size_sam_mode=size_sam_mode,
            )
        else:
            regions = sig_regions

        for sig_id, region in enumerate(regions):
            if snr_sam_mode == "lin":
                # sig_power = choice(sig_powers)
                sig_power = uniform(sig_power_range[0], sig_power_range[1])
            elif snr_sam_mode == "log":
                sig_power = uniform(np.log10(sig_power_range[0]), np.log10(sig_power_range[1]))
                sig_power = 10**sig_power
            region_shape = tuple(slice_.stop - slice_.start for slice_ in region)
            region_power = exponential(sig_power, region_shape)
            psd[region] += region_power
            if mask_mode == "binary":
                mask[region] = 1.0
            elif mask_mode == "snr":
                mask[region] += sig_power / noise_power
            elif mask_mode == "channels":
                region_m = (slice(sig_id, sig_id + 1),) + region
                mask[region_m] = 1.0

        return (psd, mask)

    def generate_psd_dataset(
        self,
        dataset_path="./data/psd_dataset.npz",
        n_dataset=1000,
        shape=(1000,),
        n_sigs_min=1,
        n_sigs_max=1,
        n_sigs_p_dist=None,
        sig_size_min=None,
        sig_size_max=None,
        snr_range=np.array([10, 10]),
        mask_mode="binary",
    ):
        self.print(
            f"Starting to generate PSD dataset with n_dataset={n_dataset}, \
                shape={shape}, n_sigs={n_sigs_min}-{n_sigs_max}, \
                    n_sigs_p_dist={n_sigs_p_dist}, sig_size={sig_size_min}-{sig_size_max}, \
                        snrs={snr_range[0]:0.3f}-{snr_range[1]:0.3f}...",
            thr=0,
        )

        n_sigs_list = np.arange(n_sigs_min, n_sigs_max + 1)
        data = []
        masks = []
        bboxes = []
        objectnesses = []
        classes = []
        for _ in range(n_dataset):
            n_sigs = np.random.choice(n_sigs_list, p=n_sigs_p_dist)
            # n_sigs = randint(n_sigs_min, n_sigs_max+1)
            regions = self.generate_random_regions(
                shape=shape,
                n_regions=n_sigs,
                min_size=sig_size_min,
                max_size=sig_size_max,
                size_sam_mode=self.config.size_sam_mode,
            )
            (psd, mask) = self.generate_random_psd(
                shape=shape,
                sig_regions=regions,
                n_sigs=n_sigs,
                n_sigs_max=n_sigs_max,
                sig_size_min=sig_size_min,
                sig_size_max=sig_size_max,
                noise_power=self.noise_power,
                snr_range=snr_range,
                size_sam_mode=self.config.size_sam_mode,
                snr_sam_mode=self.config.snr_sam_mode,
                mask_mode=mask_mode,
            )
            data.append(psd)
            masks.append(mask)
            bbox = np.zeros((n_sigs_max, 2 * len(shape)), dtype=float)
            for i, region in enumerate(regions):
                bbox[i] = np.array(
                    [slice_.start for slice_ in region]
                    + [slice_.stop - slice_.start for slice_ in region]
                )
            bbox = bbox.flatten()
            bboxes.append(bbox)
            objectness = np.array([1.0] * n_sigs + [0.0] * (n_sigs_max - n_sigs), dtype=float)
            objectnesses.append(objectness)
            class_ = np.array([0.0] * n_sigs_max, dtype=float)
            classes.append(class_)
        data = np.array(data)
        masks = np.array(masks)
        bboxes = np.array(bboxes)
        objectnesses = np.array(objectnesses)
        classes = np.array(classes)
        np.savez(
            dataset_path,
            data=data,
            masks=masks,
            bboxes=bboxes,
            objectnesses=objectnesses,
            classes=classes,
        )

        self.print(
            f"Dataset of data shape {data.shape} and mask shape {masks.shape} saved to {dataset_path}",
            thr=0,
        )

    def multi_arr_corr(self, x, arr, r, lam):
        """
        Computes the correlation between the received signal and the expected signal
        for a batch of candidate target locations at a set of arrays

        Parameters
        ----------
        x : np.array of shape (npoints,p)
            Location of the candidate targets
            where nx is the number of candidate targets and p is the
            dimension of the space
        arr : np.array  of shape (m,nrx,p)
            Array locations where arr[i,j,:] is the location
            of element j in the measurement i where m is the number of
            measurements
        r : complex np.array of size (m, nrx)
            measured values where r[i,j] is the complex measured
            value in measurement i on element j
        lam : float
            Wavelength of the signal

        Returns
        -------
        rho : np.array of shape (m)
            The real values of the summed correlation at each of the measurements
        """

        # Compute the distances from the arrays to the target
        # (:,m,nrx,p) * (npoints,:,:,p)
        d = np.sqrt(np.sum((arr[None, :, :, :] - x[:, None, None, :]) ** 2, axis=3))

        # Compute the phase difference
        dexp = np.exp(-2 * np.pi * 1j / lam * d)

        # (npoints, m, nrx)
        # Compute the correlation
        # (npoints, m)
        # (npoints)
        rho = np.sum(np.abs(np.sum(r[None, :, :] * dexp, axis=2)) ** 2, axis=1)

        return rho

    def feval_torch(self, x, arr, r, lam):
        """
        Torch version of the above function.
        Computes the correlation between the received signal and the expected signal
        for a batch of candidate target locations

        Parameters
        ----------
        x : torch.Tensor of shape (p)
            Location of the candidate targets
            where nx is the number of candidate targets and p is the
            dimension of the space
        arr : torch.Tensor of shape (m, nrx, p)
            Array locations where arr[i, j, :] is the location
            of element j in the measurement i where m is the number of
            measurements
        r : torch.Tensor of size (m, nrx)
            measured values where r[i, j] is the complex measured
            value in measurement i on element j
        lam : float
            Wavelength of the signal

        Returns
        -------
        rho : scalar
            The real values of the summed correlation at each of the measurements
        """

        # Compute the distances from the arrays to the target
        d = torch.sqrt(torch.sum((arr[:, :, :] - x[None, None, :]) ** 2, dim=2))

        # Compute the phase difference
        dexp = torch.exp(-2 * np.pi * 1j / lam * d)

        # Compute the correlation
        rho = torch.sum(torch.abs(torch.sum(r * dexp, dim=1)) ** 2, dim=0)

        return rho

    def filter_noise_symbols(self, sig, mag_thr=1e-2):
        sig_fil = sig.copy()
        sig_fil = sig_fil[np.abs(sig_fil) > mag_thr]
        # sig_fil[np.abs(sig_fil) < mag_thr] = 0

        return sig_fil


@dataclass
class SpecFilterUtilsConfig(SignalUtilsConfig):
    snr: float = None
    sig_noise: float = None
    sig_sel_id: int = None
    rx_sel_id: int = None
    n_r: int = None
    n_sig: int = None
    rand_params: dict = None
    cf_range: tuple = None
    psd_range: tuple = None
    bw_range: tuple = None
    spat_sig_range: tuple = None
    az_range: tuple = None
    el_range: tuple = None
    aoa_mode: str = None


class SpecFilterUtils(SignalUtils):
    def __init__(self, config: SpecFilterUtilsConfig, **overrides):
        super().__init__(config, **overrides)

    def gen_spatial_sig(
        self,
        n_sig=1,
        n_r=1,
        az_range=(-np.pi, np.pi),
        el_range=(-np.pi / 2, np.pi / 2),
        mode="uniform",
    ):
        ant_dim = len(self.config.ant_d)
        if ant_dim == 1:
            if mode == "uniform":
                az = uniform(az_range[0], az_range[1], n_sig)
            elif mode == "sweep":
                az_range_t = az_range[1] - az_range[0]
                az = np.linspace(az_range[0], az_range[1] - az_range_t / n_sig, n_sig)
            spatial_sig = np.exp(
                2
                * np.pi
                * 1j
                * self.config.ant_d[0]
                / self.config.wl
                * np.arange(n_r).reshape((n_r, 1))
                * np.sin(az.reshape((1, n_sig)))
            )
            return spatial_sig, [az]
        elif ant_dim == 2:
            spatial_sig = np.zeros((n_r, n_sig)).astype(complex)
            if mode == "uniform":
                az = uniform(az_range[0], az_range[1], n_sig)
                el = uniform(el_range[0], el_range[1], n_sig)
            elif mode == "sweep":
                az_range_t = az_range[1] - az_range[0]
                el_range_t = el_range[1] - el_range[0]
                az = np.linspace(az_range[0], az_range[1] - az_range_t / n_sig, n_sig)
                el = np.linspace(el_range[0], el_range[1] - el_range_t / n_sig, n_sig)
            k = 2 * np.pi / self.config.wl
            M = np.sqrt(n_r)
            N = np.sqrt(n_r)
            for i in range(n_sig):
                ax = np.exp(
                    1j * k * self.config.ant_d[0] * np.arange(M) * np.sin(el[i]) * np.cos(az[i])
                )
                ay = np.exp(
                    1j * k * self.config.ant_d[1] * np.arange(N) * np.sin(el[i]) * np.sin(az[i])
                )
                spatial_sig[:, i] = np.kron(ax, ay)
            return spatial_sig, [az, el]

    def gen_rand_params(self):
        self.print("Generating a set of random parameters.", 2)

        if self.config.rand_params:
            sig_bw = uniform(self.config.bw_range[0], self.config.bw_range[1], self.config.n_sig)
            psd_range = self.config.psd_range / 1e3 / 1e6
            sig_psd = uniform(psd_range[0], psd_range[1], self.config.n_sig)
            sig_cf = uniform(self.config.cf_range[0], self.config.cf_range[1], self.config.n_sig)

            spat_sig_mag = uniform(
                self.config.spat_sig_range[0], self.config.spat_sig_range[1], (1, self.config.n_sig)
            )
            spat_sig_mag = np.tile(spat_sig_mag, (self.config.n_r, 1))
            spatial_sig, aoa = self.gen_spatial_sig(
                n_sig=self.config.n_sig,
                n_r=self.config.n_r,
                az_range=self.config.az_range,
                el_range=self.config.el_range,
                mode=self.config.aoa_mode,
            )
            spatial_sig = spat_sig_mag * spatial_sig

        else:
            self.config.n_sig = 8
            self.config.n_r = 4
            sig_bw = np.array(
                [
                    23412323.42206957,
                    29720830.74807138,
                    28854411.42943605,
                    13436699.17479161,
                    32625455.26622169,
                    32053137.51678639,
                    35113044.93237082,
                    21712944.94126201,
                ]
            )
            sig_psd = np.array(
                [
                    1.82152663e-10 + 0.0j,
                    2.18261433e-10 + 0.0j,
                    2.10519428e-10 + 0.0j,
                    1.72903294e-10 + 0.0j,
                    2.25096120e-10 + 0.0j,
                    1.42163622e-10 + 0.0j,
                    1.16246992e-10 + 0.0j,
                    1.26733169e-10 + 0.0j,
                ]
            )
            sig_cf = np.array(
                [
                    76368431.6004079,
                    10009408.65004128,
                    -17835240.41355851,
                    -17457600.99681053,
                    -11925292.61281498,
                    36570531.45445453,
                    28089213.97482219,
                    36680162.41373056,
                ]
            )
            spatial_sig = np.array(
                [
                    [
                        0.28560148 + 0.0j,
                        0.49996994 + 0.0j,
                        0.65436809 + 0.0j,
                        0.77916855 + 0.0j,
                        0.77740179 + 0.0j,
                        0.72816271 + 0.0j,
                        0.70354769 + 0.0j,
                        0.79870358 + 0.0j,
                    ],
                    [
                        0.28247656 - 0.04213306j,
                        0.23454661 - 0.44154029j,
                        0.38195112 - 0.5313294j,
                        0.53913962 + 0.56252299j,
                        0.72772125 + 0.27345077j,
                        -0.08719831 - 0.72292281j,
                        0.30553255 - 0.63374223j,
                        0.73871532 + 0.30368912j,
                    ],
                    [
                        0.21580258 + 0.18707605j,
                        0.3849812 + 0.31899753j,
                        0.65124563 - 0.06384924j,
                        -0.75863413 - 0.17770168j,
                        0.57519734 - 0.52297377j,
                        -0.44911618 - 0.5731628j,
                        0.3280136 - 0.62240375j,
                        0.33967472 + 0.72287516j,
                    ],
                    [
                        0.24103957 + 0.1531931j,
                        0.46232038 - 0.19034129j,
                        0.32828468 - 0.56606251j,
                        -0.39663875 - 0.6706574j,
                        0.72239466 - 0.28722724j,
                        -0.51525611 + 0.51452121j,
                        -0.41820151 - 0.56576218j,
                        0.0393057 + 0.79773584j,
                    ],
                ]
            )
            aoa = None

        sig_psd = sig_psd.astype(complex)
        spatial_sig = spatial_sig.astype(complex)

        self.config.sig_bw = sig_bw
        self.config.sig_psd = sig_psd
        self.config.sig_cf = sig_cf
        self.config.spatial_sig = spatial_sig
        self.config.aoa = aoa

        return (sig_bw, sig_psd, sig_cf, spatial_sig, aoa)

    def generate_signals(self, sig_bw, sig_psd, sig_cf, spatial_sig):
        self.print("Generating a set of signals and a rx signal.", 2)

        rx = np.zeros((self.config.n_r, self.config.n_samples), dtype=complex)
        sigs = np.zeros((self.config.n_sig, self.config.n_samples), dtype=complex)

        for i in range(self.config.n_sig):
            fil_sig = firwin(1001, sig_bw[i] / self.config.fs)
            # sigs[i, :] = np.exp(2 * np.pi * 1j * sig_cf[i] * t) * sig_psd[i] * np.convolve(noise, fil_sig, mode='same')
            sigs[i, :] = (
                np.exp(2 * np.pi * 1j * sig_cf[i] * self.config.t)
                * np.sqrt(sig_psd[i] * self.config.fs / 2)
                * lfilter(fil_sig, np.array([1]), self.gen_noise(mode="complex"))
            )
            rx += np.outer(spatial_sig[:, i], sigs[i, :])

            if self.config.sig_noise:
                yvar = np.mean(np.abs(sigs[i, :]) ** 2)
                wvar = yvar / self.config.config.snr
                sigs[i, :] += np.sqrt(wvar / 2) * self.gen_noise(mode="complex")

        yvar = np.mean(np.abs(rx) ** 2, axis=1)
        wvar = yvar / self.config.snr
        self.noise_psd = np.mean(wvar / self.config.fs).astype(complex)
        noise_rx = np.array([self.gen_noise(mode="complex") for _ in range(self.config.n_r)])
        noise_rx = np.sqrt(wvar[:, None] / 2) * noise_rx
        rx += noise_rx

        if self.config.plot_level >= 2:
            plt.figure()
            # plt.figure(figsize=(10,6))
            # plt.tight_layout()
            plt.subplots_adjust(wspace=0.5, hspace=1.0)
            plt.subplot(3, 1, 1)
            for i in range(self.config.n_sig):
                spectrum = fftshift(fft(sigs[i, :]))
                spectrum = self.lin_to_db(np.abs(spectrum), mode="mag")
                plt.plot(self.config.freq, spectrum, color=rand(3), linewidth=0.5)
            plt.title("Frequency spectrum of initial wideband signals")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Magnitude (dB)")

            plt.subplot(3, 1, 2)
            spectrum = fftshift(fft(rx[self.rx_sel_id, :]))
            spectrum = self.lin_to_db(np.abs(spectrum), mode="mag")
            plt.plot(self.config.freq, spectrum, "b-", linewidth=0.5)
            plt.title("Frequency spectrum of RX signal in a selected antenna")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Magnitude (dB)")

            plt.subplot(3, 1, 3)
            spectrum = fftshift(fft(sigs[self.config.sig_sel_id, :]))
            spectrum = self.lin_to_db(np.abs(spectrum), mode="mag")
            plt.plot(self.config.freq, spectrum, "r-", linewidth=0.5)
            plt.title("Frequency spectrum of the desired wideband signal")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Magnitude (dB)")

            plt.savefig(os.path.join(self.config.figs_dir, "tx_rx_sigs.pdf"), format="pdf")
            # plt.show(block=False)

        return (rx, sigs)
