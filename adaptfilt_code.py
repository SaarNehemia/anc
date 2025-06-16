from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp
import soundfile as sf
from adaptfilt import lms, nlms, ap, nlmsru, rls

from matplotlib import use

use('TkAgg')

# spectral args
WIN_DUR = 0.064
HOP_FRAC = 0.2


def get_spectral_args(win_dur, hop_frac, fs):
    win_size = round(win_dur * fs)
    overlap = round(win_dur * fs * (1 - hop_frac))
    return win_size, overlap


def plot_psd(psd, ax):
    # ax.pcolormesh(np.arange(psd.shape[1]), np.arange(psd.shape[0]), 10 * np.log10(psd + 1e-10), shading='gouraud')
    ax.imshow(10 * np.log10(psd),
              origin='lower', aspect='auto',
              cmap='inferno',
              vmin=-90, vmax=-20
              )


def plot_psds(psds_dict: dict, kmin=0, kmax=200):
    psds_titles, psds = psds_dict.items()
    num_psds = len(psds)
    fig, ax = plt.subplots(num_psds, 1) #, sharex='all', sharey='all')
    for i, psd_title, psd in enumerate(psds_dict.items()):
        current_ax = ax[i] if num_psds > 1 else ax
        plot_psd(psd=psds[i], ax=current_ax)
        current_ax.set_ylim([kmin, kmax])
        current_ax.set_title(psd_title)

    # Get the figure manager
    # mng = plt.get_current_fig_manager()

    # Maximize the window
    # mng.window.showMaximized()  # using Qt5Agg backend (default)
    # mng.window.state('zoomed')

    # Display the plot
    plt.show()


def calc_psd(sig, fs):
    win_size, overlap = get_spectral_args(WIN_DUR, HOP_FRAC, fs)
    f, t, sig_stft = sp.stft(x=sig, fs=fs, nperseg=win_size, noverlap=overlap)
    psd = abs(sig_stft) ** 2
    return psd


def resample_sig(sig, fs, desired_fs):
    if desired_fs is not None and desired_fs != fs:
        sig = sp.resample(sig, int(len(sig) / fs * desired_fs))
        fs = desired_fs
    return sig, fs


def load_sig(sig_path: Path, start_sec=None, end_sec=None):
    if start_sec is None and end_sec is None:
        sig, fs = sf.read(file=sig_path)
    else:
        fs = sf.info(sig_path).samplerate
        sig, fs = sf.read(file=sig_path, start=int(start_sec * fs), stop=int(end_sec * fs))

    # turn to mono
    sig = sig[:, 0] if len(sig.shape) > 1 else sig

    return sig, fs


if __name__ == '__main__':
    # load sig
    sig_path = Path("speech.wav")
    sig, fs = load_sig(sig_path=sig_path, start_sec=0, end_sec=10)

    factor = 0.8
    n = factor * sig.std() * np.random.randn(len(sig))
    d = sig + n  # Speech sensor
    u = n  # Noise sensor

    N = 5
    ys = list()
    Ms = [2**i for i in range(N)]
    for i in range(5):
        # y, e, w = lms(u, d, M=Ms[i], step=0.05, leak=0.5, initCoeffs=None, N=None, returnCoeffs=False)
        y, e, w = nlms(u, d, M=Ms[i], step=0.05, eps=0.001, leak=0.5, initCoeffs=None, N=None, returnCoeffs=False)

        # y, e, w = nlmsru(u, d, M=1024, step=1, eps=0.001, leak=0, initCoeffs=None, N=None, returnCoeffs=False)
        # y, e, w = nlms(u, d, M=1024, step=1, eps=0.001, leak=0, initCoeffs=None, N=None, returnCoeffs=False)
        # y, e, w = ap(u, d, M=1024, step=1, K=1, eps=0.001, leak=0, initCoeffs=None, N=None, returnCoeffs=False)
        # y, e, w = rls(u, d, M=1024, ffactor=0.5, initCoeffs=None, N=None, returnCoeffs=False)
        ys.append(y)

    # Plot
    kmin = 0
    kmax = 200
    fig, ax = plt.subplots(N+1, 1)  # , sharex='all', sharey='all')
    psd_d = calc_psd(sig=d, fs=fs)
    plot_psd(psd=psd_d, ax=ax[0])
    ax[0].set_ylim([kmin, kmax])
    ax[0].set_title('Speech sensor (d)')
    for i in range(N):
        psd_y = calc_psd(sig=ys[i], fs=fs)
        current_ax = ax[i+1]
        plot_psd(psd=psd_y, ax=current_ax)
        current_ax.set_ylim([kmin, kmax])
        current_ax.set_title(Ms[i])
    plt.show()

    # psd_sig = calc_psd(sig=sig, fs=fs)
    # psd_y = calc_psd(sig=y, fs=fs)
    # psd_d = calc_psd(sig=d, fs=fs)
    # psd_u = calc_psd(sig=u, fs=fs)
    # plot_psds(psds_dict={'Original (sig)': psd_sig,
    #                      'Speech sensor (d)': psd_d,
    #                      'Noise sensor (u)': psd_u,
    #                      'Result (y)': psd_y})

