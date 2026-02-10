import os
import numpy as np
import scipy
from scipy.signal import lfilter, freqz
import matplotlib.pyplot as plt

from .signal_utils import Signal_Utils



# TODO add config class and transfer all parameters

class Filter_Utils(Signal_Utils):

    def __init__(self, config):
        super().__init__(config)
        

    def wiener_fir(self, input, output, filter_order_pos, filter_order_neg):

        filter_order = filter_order_pos + filter_order_neg
        filter_length = filter_order + 1
        N_samples = input.shape[1]

        rxx = np.correlate(input[0, :], input[0, :], mode='same') / N_samples
        center = len(rxx) // 2
        rxx = rxx[center:center + filter_length]
        Rxx = scipy.linalg.toeplitz(rxx)

        # Rxx_1 = np.zeros((filter_length, filter_length)).astype(complex)
        # for i in range(filter_length):
        #     for j in range(filter_length):
        #         Rxx_1[i, j] = self.cross_correlation(input[0,:], input[0,:], i - j)
        # print(np.sum(np.abs(Rxx - Rxx_1)))

        ryx = np.correlate(output[0,:], input[0,:], mode='same') / N_samples
        center = len(ryx) // 2
        ryx = ryx[center-filter_order_neg:center+filter_length-filter_order_neg]
        Ryx = np.reshape(ryx, (filter_length, 1))

        # Ryx_1 = np.zeros((filter_length, 1)).astype(complex)
        # for j in range(filter_length):
        #     Ryx_1[j, 0] = self.cross_correlation(output[0,:], input[0,:], j - filter_order_neg)
        # print(np.sum(np.abs(Ryx - Ryx_1)))

        wiener_filter_coef = np.linalg.solve(Rxx, Ryx)

        return wiener_filter_coef


    def wiener_fir_param(self, sig_id, sig_bw, sig_pwr, sig_cf, spatial_sig, snr, filter_order_pos, filter_order_neg):

        filter_order = filter_order_pos + filter_order_neg
        filter_length = filter_order + 1

        N_in = spatial_sig.shape[0]
        N_out = spatial_sig.shape[1]
        N0 = self.noise_psd
        sig_bw = sig_bw/self.fs
        sig_cf = sig_cf/self.fs

        t = np.arange(-(filter_length - 1), filter_length)
        Rxx_b1 = spatial_sig[0, :] * np.conj(spatial_sig[0, :]) * sig_bw * sig_pwr
        Rxx_b1 = Rxx_b1[:,None] * np.exp(2 * np.pi * 1j * sig_cf[:,None] * t[None,:]) * np.sinc(sig_bw[:,None] * t[None,:])
        Rxx_b2 = np.sum(Rxx_b1, axis=0)
        Rxx = scipy.linalg.toeplitz(Rxx_b2[filter_length-1:],Rxx_b2[:filter_length][::-1]) + N0 * np.eye(filter_length)

        # Rxx = np.zeros((filter_length, filter_length)).astype(complex)
        # for i in range(filter_length):
        #     for j in range(filter_length):
        #         t = i-j
        #         for k in range(N_out):
        #             Rxx[i, j] += spatial_sig[0,k] * np.conj(spatial_sig[0,k]) * np.exp(2 * np.pi * 1j * sig_cf[k] * t) * sig_pwr[k] * sig_bw[k] * np.sinc(sig_bw[k]*t)
        #         if t==0:
        #             Rxx[i, j] += N0

        t = np.arange(filter_length)
        Ryx_b1 = np.conj(spatial_sig[0, sig_id]) * sig_pwr[sig_id] * sig_bw[sig_id] * np.exp(2 * np.pi * 1j * sig_cf[sig_id] * t) * np.sinc(sig_bw[sig_id] * t)
        Ryx = Ryx_b1.reshape(-1,1)

        # Ryx = np.zeros((filter_length, 1)).astype(complex)
        # for j in range(filter_length):
        #     t = j - filter_order_neg
        #     Ryx[j, 0] = np.conj(spatial_sig[0,sig_id]) * sig_pwr[sig_id] * sig_bw[sig_id] * np.exp(2 * np.pi * 1j * sig_cf[sig_id] * t) * np.sinc(sig_bw[sig_id] * t)

        wiener_filter_coef = np.linalg.solve(Rxx, Ryx)

        return wiener_filter_coef


    def wiener_fir_vector(self, input, output, filter_order_pos, filter_order_neg):
        # input_signals is a N_in x N matrix consisting N_in signals each with N sample points
        # output_signals is N_out x N matrix consisting N_out signals each with N sample points

        filter_order = filter_order_pos + filter_order_neg
        filter_length = filter_order+1
        N_in = input.shape[0]
        N_out = output.shape[0]
        N_samples = input.shape[1]

        # start = time.time()
        Rxx_t=np.zeros((N_in,N_in,filter_length)).astype(complex)
        for i in range(N_in):
            for j in range(i,N_in):
                rxx = np.correlate(input[i,:], input[j,:], mode='same') / N_samples
                center = len(rxx) // 2
                Rxx_t[i,j,:] = rxx[center:center+filter_length]
                Rxx_t[j,i,:] = np.conj(rxx[center:center-filter_length:-1])
        Rxx = np.zeros((filter_length * N_in, filter_length * N_in)).astype(complex)
        for i in range(filter_length):
            for j in range(filter_length):
                if j>=i:
                    Rxx[i*N_in:(i+1)*N_in, j*N_in:(j+1)*N_in] = Rxx_t[:,:,j-i]
                else:
                    Rxx[i*N_in:(i+1)*N_in, j*N_in:(j+1)*N_in] = np.conj(Rxx_t[:,:,i-j]).T

        # Rxx_1 = np.zeros((filter_length * N_in, filter_length * N_in)).astype(complex)
        # for i in range(Rxx.shape[0]):
        #     for j in range(Rxx.shape[1]):
        #         idx_1 = i % N_in
        #         idx_2 = j % N_in
        #         corr_index = (j // N_in) - (i // N_in)
        #         Rxx_1[i, j] = self.cross_correlation(input[idx_1, :], input[idx_2, :], corr_index)
        # end=time.time()
        # print('1: {}'.format(end-start))
        # print(np.sum(np.abs(Rxx-Rxx_1)))

        # start = time.time()
        Ryx_t = np.zeros((N_out, N_in, filter_length)).astype(complex)
        for i in range(N_out):
            for j in range(N_in):
                ryx = np.correlate(output[i, :], input[j, :], mode='same') / N_samples
                center = len(ryx) // 2
                Ryx_t[i, j, :] = ryx[center:center + filter_length]
        Ryx = np.zeros((N_out, filter_length * N_in)).astype(complex)
        for i in range(filter_length):
            Ryx[:, i * N_in:(i + 1) * N_in] = Ryx_t[:, :, i]

        # Ryx_1 = np.zeros((N_out, filter_length * N_in)).astype(complex)
        # for i in range(Ryx.shape[0]):
        #     for j in range(Ryx.shape[1]):
        #         idx_1 = i
        #         idx_2 = j % N_in
        #         corr_index = j // N_in
        #         Ryx_1[i, j] = self.cross_correlation(output[idx_1, :], input[idx_2, :], corr_index)
        # end = time.time()
        # print('2: {}'.format(end - start))
        # print(np.sum(np.abs(Ryx - Ryx_1)))

        # self.print(f'Rxx determinant: {np.linalg.det(Rxx)}',4)

        # start = time.time()
        wiener_filter_coef = np.linalg.solve(Rxx.T, Ryx.T).T  # Equivalent to Ryx / Rxx
        # end = time.time()
        # print('3: {}'.format(end - start))

        return wiener_filter_coef


    def wiener_fir_vector_param(self, sig_bw, sig_pwr, sig_cf, spatial_sig, snr, filter_order_pos, filter_order_neg):

        # input_signals is a N_in x N matrix consisting N_in signals each with N sample points
        # output_signals is N_out x N matrix consisting N_out signals each with N sample points

        filter_order = filter_order_pos + filter_order_neg
        filter_length = filter_order+1
        N_in = spatial_sig.shape[0]
        N_out = spatial_sig.shape[1]
        # N0 = (np.sum(np.mean(spatial_sig,axis=0) * sig_pwr * sig_bw) / np.sum(sig_bw)) / snr
        N0 = self.noise_psd
        sig_bw = sig_bw / self.fs
        sig_cf = sig_cf / self.fs

        Rxx_b1 = np.zeros((N_out, N_in, N_in)).astype(complex)
        for k in range(N_out):
            Rxx_b1[k,:,:] = np.outer(spatial_sig[:, k], np.conj(spatial_sig[:, k])) * sig_pwr[k] * sig_bw[k]
        t = np.arange(-(filter_length-1),filter_length)
        Rxx_b2 = np.exp(2 * np.pi * 1j * sig_cf[:,None] * t[None,:]) * np.sinc(sig_bw[:,None] * t[None,:])
        Rxx_b3 = Rxx_b1[:,:,:,None] * Rxx_b2[:,None,None,:]
        Rxx_b3 = np.sum(Rxx_b3, axis=0)

        Rxx = np.zeros((filter_length * N_in, filter_length * N_in)).astype(complex)
        for i in range(filter_length):
            for j in range(filter_length):
                t = j-i
                Rxx[i * N_in:(i + 1) * N_in, j * N_in:(j + 1) * N_in] = Rxx_b3[:,:,t+filter_length-1] \
                                            + float(t == 0) * N0 * np.eye(N_in)
                # for k in range(N_out):
                #     Rxx[i * N_in:(i + 1) * N_in, j * N_in:(j + 1) * N_in] += Rxx_b1[k] * Rxx_b2[k,t+filter_length-1]
                #     # Rxx[i*N_in:(i+1)*N_in, j*N_in:(j+1)*N_in] += np.outer(spatial_sig[:, k], np.conj(spatial_sig[:, k])) * np.exp(2 * np.pi * 1j * sig_cf[k] * t) * sig_pwr[k] * sig_bw[k] * np.sinc(sig_bw[k] * t)


        t = np.arange(filter_length)
        Ryx_b1 = (np.conj(spatial_sig).T * sig_pwr[:,None] * sig_bw[:,None])
        Ryx_b2 = np.exp(2 * np.pi * 1j * sig_cf[:,None] * t[None,:]) * np.sinc(sig_bw[:,None] * t[None,:])

        Ryx = np.zeros((N_out, filter_length * N_in)).astype(complex)
        for j in range(filter_length):
            t = j
            Ryx_b2_t = Ryx_b2[:, t]
            Ryx[:,j*N_in:(j+1)*N_in] = Ryx_b1 * Ryx_b2_t[:, None]

            # temp = None
            # for k in range(N_out):
            #     Rx_a = Ryx_b1[k,:] * Ryx_b2[k,t].reshape(1,-1)
            #     # Rx_a = np.conj(spatial_sig[:, k]) * sig_pwr[k] * sig_bw[k] * np.exp(
            #     #     2 * np.pi * 1j * sig_cf[k] * t) * np.sinc(sig_bw[k] * t)
            #     # Rx_a = Rx_a.reshape(1,-1)
            #     if temp is None:
            #         temp = Rx_a.copy()
            #     else:
            #         temp = np.vstack((temp, Rx_a))
            # Ryx[:,j*N_in:(j+1)*N_in] = temp.copy()

        self.print(f'Rxx determinant: {np.linalg.det(Rxx)}',4)

        wiener_filter_coef = np.linalg.solve(Rxx.T, Ryx.T).T  # Equivalent to Ryx / Rxx

        return wiener_filter_coef


    def basis_fir_us(self, input, fil_base, t, freq, center_freq, iters, us_rate, plot_procedure=False):
        """
        Apply a sequence of basis FIR filters with upsampling to an input signal.

        Args:
            input (np.array): The input signal to the filter.
            fil_base (np.array): Basis filter with which the filtering is being done.
            t (np.array): The time vector.
            freq (np.array): The frequency vector.
            center_freq (float): The center frequency.
            iters (int): Number of iterations of filter upsampling.
            us_rate (int): Upsampling rate.
            plot_procedure (bool): If True, plots the process of DS, filtering, US.

        Returns:
            output (np.array): The output of the filter.
            grp_dly (int): The group delay of the filter.
        """
        om = (freq / max(freq)) * np.pi

        fil_us = [None] * iters
        sig_fil_us = [None] * iters
        fil_base_shifted = np.exp(2 * np.pi * 1j * center_freq * t[:len(fil_base)]) * fil_base

        grp_dly = ((len(fil_base)-1) // 2) * (2**(iters + 1) - 1)

        if iters == 0:
            # output = np.convolve(input, fil_base_shifted, mode='same')
            output = lfilter(fil_base_shifted, np.array([1]), input)
        else:
            temp = fil_base.copy()
            for i in range(iters):
                # fil_us[i] = upfirdn([1], temp, up=us_rate)
                fil_us[i] = self.upsample(temp, up=us_rate)
                temp = fil_us[i]
            for i in range(iters):
                fil_us[i] = np.exp(2 * np.pi * 1j * center_freq * t[:len(fil_us[i])]) * fil_us[i]

            temp = input.copy()
            for i in range(iters):
                # sig_fil_us[i] = np.convolve(temp, fil_us[iters - i - 1], mode='same')
                sig_fil_us[i] = lfilter(fil_us[iters - i - 1], np.array([1]), temp)
                temp = sig_fil_us[i]
            # output = np.convolve(temp, fil_base_shifted, mode='same')
            output = lfilter(fil_base_shifted, np.array([1]), temp)

            if plot_procedure:
                # Plotting the filter response and signal spectrums
                plt.figure()
                w, h = freqz(fil_base_shifted, worN=om)
                plt.plot(w/np.pi, self.lin_to_db(abs(h), mode='mag'), linewidth=1.0, label='Base Filter')
                w, h = freqz(fil_us[0], worN=om)
                plt.plot(w/np.pi, self.lin_to_db(abs(h), mode='mag'), linewidth=1.0, label='Upsampled Filter')
                plt.title('Base and Upsampled Filters Frequency Response')
                plt.xlabel(r'Normalized Frequency ($\times \pi$ rad/sample)')
                plt.ylabel('Magnitude (dB)')
                plt.legend()
                # plt.show()

                plt.figure()
                plot_indices = [0,min(1,iters-1),iters-1]
                # for idx, signal in enumerate(sig_fil_us, start=1):
                for i, idx in enumerate(plot_indices):
                    plt.subplot(len(plot_indices) + 1, 1, i+1)
                    spectrum = np.fft.fftshift(np.fft.fft(sig_fil_us[idx]))
                    plt.plot(freq, self.lin_to_db(np.abs(spectrum), mode='mag'), linewidth=1.0)
                    plt.title(f'Frequency Spectrum of the {idx} round filtered signal')
                    plt.xlabel('Frequency (Hz)')
                    plt.ylabel('Magnitude (dB)')

                plt.subplot(len(plot_indices)+1, 1, len(plot_indices)+1)
                spectrum = np.fft.fftshift(np.fft.fft(output))
                plt.plot(freq, self.lin_to_db(np.abs(spectrum), mode='mag'), linewidth=1.0)
                plt.title('Frequency Spectrum of the output signal')
                plt.xlabel('Frequency (Hz)')
                plt.ylabel('Magnitude (dB)')
                # plt.show()

        # output = np.array(output)
        return output, grp_dly



    def basis_fir_ds_us(self, input, fil_base, t, freq, center_freq, iters, ds_rate, us_rate, plot_procedure=False):
        """
        Apply a basis FIR filter with downsampling and upsampling to an input signal.

        Args:
            input (np.array): The input signal to the filter.
            fil_base (np.array): Basis filter with which the filtering is being done.
            t (np.array): The time vector.
            freq (np.array): The frequency vector.
            center_freq (float): The center frequency.
            iters (int): Number of iterations of downsampling and upsampling.
            ds_rate (int): Downsampling rate.
            us_rate (int): Upsampling rate.
            plot_procedure (bool): If True, plots the process of DS, filtering, US.

        Returns:
            output (np.array): The output of the filter.
            grp_dly (int): The group delay of the filter.
        """
        om = (freq / max(freq)) * np.pi

        sig_ds = [None] * iters
        sig_us = [None] * iters
        sig_ds_fil = [None] * iters
        sig_us_fil = [None] * iters
        fil_base_shifted = np.exp(2 * np.pi * 1j * center_freq * t[:len(fil_base)]) * fil_base
        input_centered = np.exp(-2 * np.pi * 1j * center_freq * t) * input

        grp_dly = ((len(fil_base)-1) // 2) * (3 * (2 ** iters) - 2)
        if iters == 0:
            # output = np.convolve(input, fil_base_shifted, mode='same')
            output = lfilter(fil_base_shifted, np.array([1]), input)
        else:
            temp = input_centered.copy()
            for i in range(iters):
                # sig_ds_fil[i] = np.convolve(temp, fil_base, mode='same')
                sig_ds_fil[i] = lfilter(fil_base, np.array([1]), temp)
                sig_ds[i] = sig_ds_fil[i][::ds_rate]
                temp = sig_ds[i]

            # temp = np.convolve(temp, fil_base, mode='same')
            temp = lfilter(fil_base, np.array([1]), temp)
            for i in range(iters):
                # sig_us[i] = upfirdn([1], temp, up=us_rate)
                sig_us[i] = self.upsample(temp, up=us_rate)
                # sig_us_fil[i] = np.convolve(sig_us[i], fil_base, mode='same')
                sig_us_fil[i] = lfilter(fil_base, np.array([1]), sig_us[i])
                temp = sig_us_fil[i]

            output = sig_us_fil[iters - 1] * (2 ** iters)
            output = np.exp(2 * np.pi * 1j * center_freq * t) * output

            if plot_procedure:
                # Plotting the filter response and signal spectrums
                plt.figure()
                for idx in range(4):
                    plt.subplot(4, 1, idx + 1)
                    if idx == 0:
                        spectrum = np.fft.fftshift(np.fft.fft(sig_ds_fil[0]))
                        index = 0
                        plt.title('Frequency spectrum of the first round filtered signal')
                    elif idx == 1:
                        spectrum = np.fft.fftshift(np.fft.fft(sig_ds[0]))
                        index = 1
                        plt.title('Frequency spectrum of the first round downsampled filtered signal')
                    elif idx == 2:
                        spectrum = np.fft.fftshift(np.fft.fft(sig_ds_fil[iters - 1]))
                        index = iters-1
                        plt.title('Frequency spectrum of the last round filtered signal')
                    elif idx == 3:
                        spectrum = np.fft.fftshift(np.fft.fft(sig_ds[iters - 1]))
                        index = iters
                        plt.title('Frequency spectrum of the last round downsampled filtered signal')
                    spectrum = self.lin_to_db(np.abs(spectrum), mode='mag')
                    freq_ds = freq[::ds_rate**index]
                    plt.plot(freq_ds, spectrum, 'r-', linewidth=1.0)
                    plt.xlabel('Frequency (Hz)')
                    plt.ylabel('Magnitude (dB)')

                plt.figure()
                for idx in range(4):
                    plt.subplot(4, 1, idx + 1)
                    if idx == 0:
                        spectrum = np.fft.fftshift(np.fft.fft(sig_us[0]))
                        index = 1
                        plt.title('Frequency spectrum of the first round upsampled signal')
                    elif idx == 1:
                        spectrum = np.fft.fftshift(np.fft.fft(sig_us_fil[0]))
                        index = 1
                        plt.title('Frequency spectrum of the first round filtered upsampled signal')
                    elif idx == 2:
                        spectrum = np.fft.fftshift(np.fft.fft(sig_us[iters - 1]))
                        index = iters
                        plt.title('Frequency spectrum of the last round upsampled signal')
                    elif idx == 3:
                        spectrum = np.fft.fftshift(np.fft.fft(sig_us_fil[iters - 1]))
                        index = iters
                        plt.title('Frequency spectrum of the last round filtered upsampled signal')
                    spectrum = self.lin_to_db(np.abs(spectrum), mode='mag')
                    freq_us = freq[::us_rate ** (iters - index)]
                    plt.plot(freq_us, spectrum, 'r-', linewidth=1.0)
                    plt.xlabel('Frequency (Hz)')
                    plt.ylabel('Magnitude (dB)')

                # plt.show()

        return output, grp_dly


    def wiener_filter_design(self, mode='params', rx=None, sigs=None, sig_bw=None, sig_psd=None, sig_cf=None, spatial_sig=None):
        self.print('Beginning to design the optimal wiener filter in mode: {}'.format(mode),2)

        self.fil_wiener_single = [[None] * self.N_r for _ in range(self.N_sig)]
        if self.N_r <= 1:
        # if self.N_r <= 0:
            for i in range(self.N_sig):
                for j in range(self.N_r):
                    if mode=='sigs':
                        self.fil_wiener_single[i][j] = self.wiener_fir(rx, sigs[i, :].reshape((1, -1)), self.wiener_order_pos,
                                                                             self.wiener_order_neg).reshape(-1)
                    elif mode=='params':
                        self.fil_wiener_single[i][j] = self.wiener_fir_param(i, sig_bw, sig_psd, sig_cf, spatial_sig,
                                                                             self.snr, self.wiener_order_pos,
                                                                             self.wiener_order_neg).reshape(-1)
        else:
            if mode=='sigs':
                fil_wiener = self.wiener_fir_vector(rx, sigs, self.wiener_order_pos, self.wiener_order_neg)
            elif mode=='params':
                fil_wiener = self.wiener_fir_vector_param(sig_bw, sig_psd, sig_cf, spatial_sig, self.snr,
                                                      self.wiener_order_pos, self.wiener_order_neg)
            for i in range(self.N_sig):
                for j in range(self.N_r):
                    self.fil_wiener_single[i][j] = fil_wiener[i, j::self.N_r]

        if self.plot_level >= 3:
            # plt.figure()
            # w, h = freqz(self.fil_wiener_single[self.sig_sel_id][self.rx_sel_id], worN=om)
            # plt.plot(w / np.pi, self.lin_to_db(np.abs(h), mode='mag'), linewidth=1.0)
            # plt.title('Frequency response of the Wiener filter \n for the selected TX signal and RX antenna')
            # plt.xlabel(r'Normalized Frequency ($\times \pi$ rad/sample)')
            # plt.ylabel('Magnitude (dB)')
            # # plt.show(block=False)

            plt.figure()
            plt.subplots_adjust(wspace=0.5, hspace=1.0)
            for rx_id in range(self.N_r):
                plt.subplot(self.N_r,1,rx_id+1)
                w, h = freqz(self.fil_wiener_single[self.sig_sel_id][rx_id], worN=self.om)
                plt.plot(w / np.pi, self.lin_to_db(np.abs(h), mode='mag'), linewidth=1.0)
                plt.title('Selected TX signal, and RX antenna {}'.format(rx_id+1))
                if rx_id == 1:
                    plt.ylabel('Magnitude (dB)')
            plt.xlabel(r'Normalized Frequency ($\times \pi$ rad/sample)')
            plt.savefig(os.path.join(self.figs_dir, 'wiener_filters.pdf'), format='pdf')
            # plt.show(block=False)


        # if self.plot_level>=1:
        #     print('bw: ',self.sig_bw/1e6)
        #     print('cf: ',self.sig_cf/1e6)
        #     print('aoa: ',self.aoa)
        #     f_len = len(self.freq)
        #     S = np.zeros((self.N_sig, f_len))
        #     for sig_id in range(self.N_sig):
        #         h_simo = np.zeros((self.N_r, f_len), dtype=complex)
        #         for rx_id in range(self.N_r):
        #             w, h = freqz(self.fil_wiener_single[sig_id][rx_id], worN=self.om)
        #             h_simo[rx_id,:] = h
        #         # print(np.mean(np.abs(h_simo)))
        #         # print(np.mean(np.abs(self.spatial_sig[:,sig_id])))
        #         S[sig_id,:] = np.abs(np.dot(np.conj(self.spatial_sig[:,sig_id].T), h_simo))**2
        #         # print(np.mean(np.abs(S[sig_id,:])))
        #     # print(np.mean(np.abs(S), axis=1))
        #     # print(np.abs(S))
        #
        #     S = self.lin_to_db(S)
        #     plt.figure(figsize=(10, 6))
        #     plt.imshow(S, aspect='auto',
        #                # extent=[self.freq[0] / 1e6, self.freq[-1] / 1e6, self.aoa[0][-1], self.aoa[0][0]],
        #                extent=[self.freq[0] / 1e6, self.freq[-1] / 1e6, -np.pi, np.pi],
        #                cmap='viridis', interpolation='nearest')
        #     plt.colorbar(label=r'Gain $|S(\theta,f)|^2$')
        #     plt.xlabel('Frequency (MHz)')
        #     plt.ylabel('Angle of Arrival (Rad)')
        #     plt.title('Wiener Filter Response Heatmap')
        #     plt.show()

        return self.fil_wiener_single


    def wiener_filter_apply(self, rx, sigs):
        self.print('Beginning to apply the optimal wiener filter on the rx signal.',2)

        rx_dly = rx.copy()
        self.wiener_errs = np.zeros(self.N_sig)

        for i in range(self.N_sig):
            sig_filtered_wiener = np.zeros_like(self.t, dtype=complex)
            for j in range(self.N_r):
                # sig_filtered_wiener += np.convolve(rx_dly[j, :], fil_wiener_single[i][j], mode='same')
                sig_filtered_wiener += lfilter(self.fil_wiener_single[i][j], np.array([1]), rx_dly[j, :])

            time_delay = self.extract_delay(sig_filtered_wiener, sigs[i, :], self.plot_level >= 5)
            self.print(f'Time delay between the signal and its Wiener filtered version for signal {i + 1}: {time_delay} samples',3)

            sig_filtered_wiener_adj, signal_adj, mse, err2sig_ratio = self.time_adjust(sig_filtered_wiener, sigs[i, :],
                                                                                             time_delay)
            self.print(
                f'Error to signal ratio for the estimation of the main signal using Wiener filter for signal {i + 1}: {err2sig_ratio}',3)
            self.wiener_errs[i] = err2sig_ratio

            if i == self.sig_sel_id and self.plot_level >= 4:
                plt.figure()
                index = range(self.n_samples // 2, self.n_samples // 2 + 500)
                plt.plot(self.t[index], np.abs(signal_adj[index]), 'r-', linewidth=0.5)
                plt.plot(self.t[index], np.abs(sig_filtered_wiener_adj[index]), 'b-', linewidth=0.5)
                plt.title('Signal and its recovered wiener filtered in time domain')
                plt.xlabel('Time(s)')
                plt.ylabel('Magnitude')
                # plt.show(block=False)


if __name__ == '__main__':
    pass



