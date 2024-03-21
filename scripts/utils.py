import math
import numpy as np
import os
from typing import Tuple


def time2idx(t: np.array, sr: int) -> np.array:
    r""" converts time (in sec) to index values """
    return np.array([int(np.floor(t[0] * sr)),
                     int(np.ceil(t[1] * sr))])


def stm(x: str) -> str:
    r""" returns file name stem """
    return os.path.splitext(os.path.basename(x))[0]


def opt_default(c: dict, d: dict) -> dict:
    r""" recursively adds default fields of dict d to dict c
    if not yet specified in c
    Arguments:
        c: (dict) someDict
        d: (dict) defaultDict
    Returns:
        c: (dict) mergedDict; defaults added to c
    """
    for x in d:
        if x not in c:
            c[x] = d[x]
        elif type(c[x]) is dict:
            c[x] = opt_default(c[x], d[x])
    return c


def idx2sec(i: int, fs: int, ons: int = 0) -> float:

    '''
    transforms numpy array indices (=samplesIdx-1) to seconds
    
    Args:
    i: (int) index
    fs: (int) sampling rate
    ons: (int) onset index

    Returns:
    s: (float) time in seconds

    '''
    
    return (i + 1 + ons) / fs


def rmsd(x: np.array, y: np.array = None) -> float:

    '''
    returns RMSD of two vectors x and y
    
    Args:
       x: (np.array)
       y: (np.array) <zeros(len(x))>
    
    Returns:
       (float) root mean squared dev between x and y
    '''
    
    if y is None:
        y = np.zeros(len(x))

    x = np.array(x)
    y = np.array(y)
        
    return np.sqrt(np.mean((x - y) ** 2))


def windowing_idx(i: int, s: dict) -> np.array:

    '''
    as windowing(), but returning all indices from onset to offset

    Args:
        i: (int) current index
        s: (dict)
            .win window length
            .rng [on, off] range of indices to be windowed
    
    Returns:
        (np.array) [on:1:off] in window around i

    '''

    on, off = windowing(i, s)
    return np.arange(on, off, 1)

def windowing(i: np.array, o: dict) -> Tuple[int, int]:

    '''
    window of length o["win"] on and offset around single index
    limited by range o["rng"]. vectorized version: seq_windowing()
    
    Args:
        i: (array) indices
        o: (dict)
            win: (int) window length in samples
            rng: (list) [on, off] range of indices to be windowed
    
    Returns:
        on, off (int-s) indices of window around i
    '''

    win = o["win"]
    rng = o["rng"]
    
    # half window
    wl = max([1, math.floor(win / 2)])
    on = max([rng[0], i - wl])
    off = min([i + wl, rng[1]])
    
    # extend window
    d = (2 * wl - 1) - (off - on)
    if d > 0:
        if on > rng[0]:
            on = max([rng[0], on - d])
        elif off < rng[1]:
            off = min([off + d, rng[1]])

    return on, off
