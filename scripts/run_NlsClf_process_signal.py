import numpy as np
import os
import sys
import torch

cwd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(cwd, "..")))

from nonlex_speech_clf import NlsClf

sr = 16000
sig = np.random.rand(sr)

nls = NlsClf()
y = nls.process_signal(sig, sr)
print(y)

