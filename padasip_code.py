import scipy.signal as sp
import soundfile as sf
import padasip as pa

import matplotlib.pyplot as plt
import numpy as np

# spectral args
WIN_DUR = 0.064
HOP_FRAC = 0.2


def get_spectral_args(win_dur, hop_frac, fs):
    win_size = round(win_dur * fs)
    overlap = round(win_dur * fs * (1 - hop_frac))
    return win_size, overlap


def calc_psd(sig, fs):
    win_size, overlap = get_spectral_args(WIN_DUR, HOP_FRAC, fs)
    f, t, sig_stft = sp.stft(x=sig, fs=fs, nperseg=win_size, noverlap=overlap)
    psd = abs(sig_stft) ** 2
    return psd


def plot_spectrogram(signal, sr, title):
    psd = calc_psd(sig=signal, fs=sr)
    plt.imshow(10 * np.log10(psd + 1e-10), aspect='auto', origin='lower')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title(title)
    plt.ylim([0, 100])
    plt.colorbar(label='dB')


if __name__ == '__main__':
    # Load clean speech
    speech, sr = sf.read("speech.wav")
    if speech.ndim > 1:
        speech = speech.mean(axis=1)

    # Simulate noise
    load_noise = True
    if load_noise:
        noise, _ = sf.read("car-street-noise-68104.wav")

        # Convert to mono if needed
        if speech.ndim > 1:
            speech = speech.mean(axis=1)
        if noise.ndim > 1:
            noise = noise.mean(axis=1)

        reference_noise = 20 * noise
    else:
        np.random.seed(42)
        noise_std = 1e3 * speech.std()
        noise = np.random.normal(0, noise_std, size=len(speech))
        reference_noise = noise
        # reference_noise = np.convolve(noise, np.ones(200) / 200, mode='same')  # blurred

    # Trim
    min_len = min(len(speech), len(reference_noise))
    speech = speech[:min_len]
    noise_start = int(len(reference_noise) / 2)
    reference_noise = reference_noise[noise_start:noise_start + min_len]

    # Mix speech + noise
    primary_signal = speech + reference_noise
    # print(f'{speech.std()=}, {reference_noise.std()=}, {primary_signal.std()=}')

    # Adaptive Filter
    filter_length = 128
    mu = 0.05
    X = pa.preprocess.input_from_history(reference_noise, filter_length)
    d = primary_signal[filter_length - 1:]  # ← FIXED LENGTH

    # Truncate if needed (safe fix)
    min_len = min(len(X), len(d))
    X = X[:min_len]
    d = d[:min_len]

    # LMS filtering
    filt = pa.filters.FilterLMS(n=filter_length, mu=mu)
    y, e, w = filt.run(x=X, d=d)

    # Save and plot
    sf.write('noisy_speech.wav', primary_signal, sr)
    sf.write('noise.wav', reference_noise, sr)
    sf.write("enhanced_simulated.wav", e, sr)

    # Plot
    plt.figure(figsize=(12, 9))
    plt.subplot(3, 1, 1)
    plot_spectrogram(primary_signal, sr, "Speech + Simulated Noise")
    plt.subplot(3, 1, 2)
    plot_spectrogram(reference_noise, sr, "Reference Noise")
    plt.subplot(3, 1, 3)
    plot_spectrogram(e, sr, "Enhanced Output")
    plt.tight_layout()
    plt.show()
