import numpy as np
import soundfile as sf

import adaptfilt_code

sig = np.load('speech.npy')
sig, fs = anc.resample_sig(sig, fs=44100/4, desired_fs=44100)
sf.write(file='speech.wav',data=sig/np.max(sig), samplerate=fs)